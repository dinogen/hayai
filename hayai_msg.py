# send a telegram message to a user
from telethon import TelegramClient
import hayai_util as util

def send_message(msg:str):
    api_id = util.context['telegram_api_id']
    api_hash = util.context['telegram_api_hash']
    bot_token = util.context['telegram_bot_token']
    chat_id = int(util.context['telegram_chat_id'])
    with TelegramClient('bot', api_id, api_hash).start(bot_token=bot_token) as client:
        client.loop.run_until_complete(client.send_message(chat_id, msg))

def send_file(file_path:str, caption:str = ''):
    api_id = util.context['telegram_api_id']
    api_hash = util.context['telegram_api_hash']
    bot_token = util.context['telegram_bot_token']
    chat_id = int(util.context['telegram_chat_id'])
    with TelegramClient('bot', api_id, api_hash).start(bot_token=bot_token) as client:
        client.loop.run_until_complete(client.send_file(chat_id, file_path, caption=caption))

if __name__ == '__main__':
    util.create_context('model')
    send_message('Hello, this is a test message from the Telegram bot!')
    send_file('requirements.txt', caption='This is a test file sent from the Telegram bot.')