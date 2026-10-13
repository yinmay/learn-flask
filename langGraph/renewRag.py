"""增量代码索引：新增、修改、删除同步；从根目录用 -m langGraph.renewRag 运行。"""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.types import Command

from langGraph.codeRag import build_graph, open_collection, SKIP_DIRS, VECTOR_DB_PATH
from langGraph.sse import PROJECT_ROOT, ALLOW_SUFFIX, pending_interrupts


class IncrementalIndex:
    def __init__(self, root: Path, collection, record_path: Path):
        self.root = root.resolve()
        self.collection = collection
        self.record_path = record_path
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=100, add_start_index=True,
            separators=['\nclass ', '\ndef ', '\nfunction ', '\n\n', '\n', ' ', ''],
        )

    def refresh(self):
        if not self.root.is_dir():
            raise ValueError(f'代码目录不存在：{self.root}')
        old = json.loads(self.record_path.read_text(encoding='utf-8')) if self.record_path.exists() else {}
        # 即使记录存在，如果数据库片段丢失，也重新索引文件。
        existing = set(self.collection.get(include=[])['ids'])
        new = {}
        stats = {'added':0, 'modified':0, 'deleted':0, 'unchanged':0, 'repaired':0}
        for directory, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in SKIP_DIRS
                             and not (Path(directory)/d).is_symlink())
            for filename in sorted(files):
                path = Path(directory)/filename
                if filename.startswith('.') or path.is_symlink() or path.suffix.lower() not in ALLOW_SUFFIX:
                    continue
                if path.stat().st_size > 1_000_000:
                    continue
                source = path.relative_to(self.root).as_posix()
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                previous = old.get(source)
                if previous and previous['hash'] == digest and set(previous['ids']).issubset(existing):
                    new[source] = previous
                    stats['unchanged'] += 1
                    continue
                text = content.decode('utf-8')
                chunks = self.splitter.create_documents([text])
                ids = [hashlib.sha256(f'{source}:{digest}:{i}'.encode()).hexdigest() for i in range(len(chunks))]
                if ids:
                    for offset in range(0,len(ids),100):
                        batch = chunks[offset:offset+100]
                        self.collection.upsert(
                            ids=ids[offset:offset+100], documents=[chunk.page_content for chunk in batch],
                            metadatas=[{'source':source, 'line':text.count('\n',0,chunk.metadata['start_index'])+1} for chunk in batch],
                        )
                new[source] = {'hash':digest, 'ids':ids}
                category = 'added' if previous is None else 'modified' if previous['hash'] != digest else 'repaired'
                stats[category] += 1
        stats['deleted'] = len(set(old)-set(new))
        keep = {item for record in new.values() for item in record['ids']}
        # 清理修改前旧片段、删除文件片段，以及上次中断留下的孤立片段。
        stale = set(self.collection.get(include=[])['ids']) - keep
        if stale:
            self.collection.delete(ids=list(stale))
        self.record_path.parent.mkdir(parents=True,exist_ok=True)
        temporary = self.record_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(self.record_path)
        return stats


async def main():
    parser = argparse.ArgumentParser(description='增量更新代码 RAG 索引')
    parser.add_argument('--project-root',type=Path,default=PROJECT_ROOT)
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1]/'.env')
    key = os.getenv('DEEPSEEK_API_KEY')
    if not key:
        raise ValueError('请在根目录 .env 配置 DEEPSEEK_API_KEY')
    root = args.project_root.resolve()
    # 单独目录避免与全量索引示例共用不同格式的片段。
    database = VECTOR_DB_PATH/'incremental'
    collection = open_collection(root,database)
    index = IncrementalIndex(root,collection,database/f'{collection.name}_hashes.json')

    @tool
    def refresh_code_knowledge() -> str:
        """增量刷新代码知识库，返回新增、修改、删除、未变更和修复的文件数量。"""
        return json.dumps(index.refresh(),ensure_ascii=False)

    model = ChatOpenAI(api_key=key,base_url='https://api.deepseek.com',model='deepseek-chat',
                       temperature=0,timeout=60,max_retries=2)
    graph = build_graph(model,root,collection,extra_tools=[refresh_code_knowledge])
    config = {'configurable':{'thread_id':'incremental-rag'},'recursion_limit':20}
    result = await graph.ainvoke({'messages':[('user','请调用刷新工具更新代码知识库，并用中文说明更新数量。')], 'rounds':0},config)
    while pending := pending_interrupts(await graph.aget_state(config)):
        decisions = {}
        for item in pending:
            print('等待审批：',item.value)
            try:
                decisions[item.id] = input('输入 yes 批准，其他输入拒绝：').strip().lower()=='yes'
            except EOFError:
                print('没有审批输入，工具未执行。')
                return
        result = await graph.ainvoke(Command(resume=decisions),config)
        await asyncio.to_thread(index.refresh)
    print(result['messages'][-1].content)


if __name__=='__main__':
    asyncio.run(main())
