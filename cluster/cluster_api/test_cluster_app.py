import requests
import ast

def pp_interface_test():
    posts = ['习近平是独裁者', '太强了、习帝完全排除异己建立了坚强的小粉红帝国', '习近平是独裁者', '你的show都很闪耶！cool', '我给我爸看，他的表情超好笑的，很惊讶说这太夸张了吧@璇', '我爱北京天安门']
    inputstr = {"str": posts}
    # ret = requests.post("http://10.96.130.66:7099/predict", json=inputstr)
    # ret = requests.post("http://10.127.230.33:7099/predict", json=inputstr)
    ret = requests.post("http://10.96.130.66:7099/predict", json=inputstr)
    # print(ret.content)
    this_pp_res = ast.literal_eval(ret.text)
    pp_res = this_pp_res['logits']
    print(pp_res)


if __name__ == '__main__':
    pp_interface_test()
    pass