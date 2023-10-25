import os
from dotenv import load_dotenv
from flask import Flask, request

from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler

from lib import utils


# .envの読み込み
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv()

# Flaskの設定
app = Flask(__name__)

# Slack Appの設定
slack_app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)
handler = SlackRequestHandler(slack_app)

@app.route("/", methods=["POST"])
def slack_event():
    return handler.handle(request)

# deeplakeにDBを作成するエンドポイント
@app.route('/embedder')
def embedder():
    _embedder = utils.Embedder(os.environ.get("GITHUB_URL"), os.environ.get("DEEPLAKE_PATH"), os.environ.get("CLONE_PATH"))
    _embedder.clone_repo()
    _embedder.create_db()

    return 'embeded completed'

# Slackのメンション時のイベントリスナー
@slack_app.event("app_mention")
def talker(body, say):
    print(body["event"]["text"])

    # 入力されたテキストとスレッドを取得
    prompt = body["event"]["text"]
    thread_ts = body["event"].get("thread_ts", None) or body["event"]["ts"]

    # ChatGPTからのレスポンス取得
    _talker = utils.Talker(os.environ.get("DEEPLAKE_PATH"))
    _talker.load_db()

    response = _talker.retrieve_results(prompt)

    # 文字列を整形
    response = response.replace('。','。\n')

    # Slackに返答を送信
    say(text=response, thread_ts=thread_ts)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))