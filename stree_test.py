import time
import random
from locust import HttpUser, task, between

class GradioQueueUser(HttpUser):
    host = "http://172.16.20.163:7860"
    wait_time = between(1, 2) #任务停留时间，休息1-2秒再发

    def on_start(self):
        # 使用前端抓到的会话参数（若动态可改为抓包获取）
        self.session_hash = "u9s8qk3c0c"  # 请替换为你的实际 session_hash
        self.trigger_id = 6               # 请替换为你的实际 trigger_id

    @task
    def ask_question(self): 
        # 随机生成加法问题
        # a = random.randint(1, 100)
        # b = random.randint(1, 100)
        # question = f"{a} + {b} 等于多少？"
        questions = [
        "巷道断面积为5㎡，高压喷雾降尘的喷雾流量应大于等于多少",
        "采煤工作面喷雾流量标准是多少？"
    ] 
        question = random.choice(questions)


        # 1) 提交 (join)，携带 trigger_id 和 session_hash
        join_payload = { 
            #"data": [question, [[question, ""]]], 注意 我gradio接口有一个事件处理函数 需要两个输入（handle_submit_with_thingking）  下面的表示没有历史对话，这个表示有历史对话，但是ai没回答，建议下面的
            "data": [question,[]],
            "fn_index": 1,
            "event_data": None,
            "session_hash": self.session_hash,
            "trigger_id": self.trigger_id
        }
        with self.client.post( #发起入队请求
            "/gradio_api/queue/join",
            json=join_payload,
            name="Join",
            catch_response=True #手动标记任务的成功和失败
        ) as join_resp:
            if join_resp.status_code != 200:
                join_resp.failure(f"Join 请求失败: {join_resp.status_code}")
                return
            resp_json = join_resp.json() #获取event_id 用于后序轮询
            event_id = resp_json.get("event_id")
            if not event_id:
                join_resp.failure("未获取到 event_id")
                return

        # 2) 轮询 (GET /queue/data) 这个相当于，你发起一次提问，任务先排队，然后你没过多少秒去问任务完成没，循环问指定次数
        for _ in range(20):
            url = ( #构造get url
                f"/gradio_api/queue/data"
                f"?session_hash={self.session_hash}&event_id={event_id}" #gradio接口在3.x版本就没有event_id   实际上还是需要event_id用于表示查询任务
                #f"?session_hash={self.session_hash}"
            )
            with self.client.get(url, name="Poll", catch_response=True) as poll_resp: #发送轮询请求
                if poll_resp.status_code != 200:
                    poll_resp.failure(f"轮询失败: {poll_resp.status_code}")
                    return
                try:
                    body = poll_resp.json()
                except Exception:
                    poll_resp.failure("解析 JSON 失败")
                    return
                if "data" in body:
                    return
                else:
                    poll_resp.success()
            time.sleep(0.5)

        # 超时
        raise Exception("轮询超时，未收到 data 字段")
