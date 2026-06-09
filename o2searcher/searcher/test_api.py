import requests
import json

def test_closed_ended_gateway():
    print("\n" + "="*50)
    print("🚀 测试 10001 端口：闭合题维基检索网关 (Closed-Ended)")
    print("="*50)
    
    url = "http://127.0.0.1:10001/wiki_search"
    
    # 修复：10001 网关严格要求二维嵌套结构 List[List[str]]
    payload = {
        "queries": [
            ["when did Ford stop producing the 7.3 diesel?"],
            ["who sang When you're in love with a beautiful woman"]
        ]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        print(f"📡 请求状态码 (Status Code): {response.status_code}")
        
        if response.status_code == 200:
            print("🟢 闭合域检索成功！返回的数据样例：")
            results = response.json()
            print(json.dumps(results, indent=4, ensure_ascii=False)[:800] + "\n...(后方文档过多，截断显示)...")
        else:
            print(f"❌ 检索失败，返回错误码 {response.status_code}，详情: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败：[Errno 111] Connection refused。请先确保一键启动脚本正常跑完。")


def test_open_ended_gateway():
    print("\n" + "="*50)
    print("🚀 测试 10102 端口：开放题长考网关 (Open-Ended)")
    print("="*50)
    
    url = "http://127.0.0.1:10102/search"
    
    # 开放题网关要求 List[List[str]]，与 closed-ended 网关保持一致，
    # 否则 server 端会按字符拆 query，导致一次返回几十条结果。
    payload = {
        "queries": [
            ["Principles and methodologies of explainable artificial intelligence (XAI) in healthcare"],
            ["Challenges and limitations of implementing XAI in medical diagnosis"]
        ]
    }
    
    try:
        # 修复：将 timeout 放大到 120 秒，容忍 DeepSeek 接口的长考总结耗时
        print("📡 正在向 10102 发送开放式调研（正在调用 DeepSeek 做长文档 RAG，可能需要 30-60 秒，请耐心等待）...")
        response = requests.post(url, json=payload, timeout=120)
        print(f"📡 请求状态码 (Status Code): {response.status_code}")
        
        if response.status_code == 200:
            print("🟢 开放域长考总结成功！DeepSeek 提炼出的 Learnings 如下：")
            results = response.json()
            print(json.dumps(results, indent=4, ensure_ascii=False))
        else:
            print(f"❌ 检索失败，返回错误码 {response.status_code}，详情: {response.text}")
            
    except requests.exceptions.ReadTimeout:
        print("❌ 错误：DeepSeek API 响应严重超时！请检查是否网络抖动，或者在 config.json 里检查你的 API Key 是否有效。")
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败：[Errno 111] Connection refused。")

if __name__ == "__main__":
    test_closed_ended_gateway()
    test_open_ended_gateway()