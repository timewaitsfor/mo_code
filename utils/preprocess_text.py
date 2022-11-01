import re
import w3lib.html
import zhconv

def filter_illegal_char(text, REPLACE_CHAR=''):
    # u0800-u4e00 (日文)
    # u4e00-u9fa5 (中文)
    # uac00-ud7a3 (韩文)
    punctuation = r'。，、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑•¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫々' \
                  r'﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎＋+×=<＿_-\ˇ~﹉﹊（）<>〈〉‹›﹛﹜『』〖〗［］《》' \
                  r'〔〕{}①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㈠㈡㈢㈣<>㈤㈥㈦㈧㈨㈩⑴⑵⑶⑷⑸⑹⑺⑻⑼' \
                  r'⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ' \
                  r'ⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ‧·・●【】–─' \
                  r'〇０１２３４５６８７８９'
    _text_buf = ''
    for ch in text:
        n = ord(ch)
        if (20 <= n < 127) or (0x4e00 <= n <= 0x9fa5) or (ord('Ａ') <= n <= ord('Ｚ')) or (ord('ａ') <= n <= ord('ｚ')):
            _text_buf += ch
        elif ch in punctuation:
            _text_buf += ch
        else:
            _text_buf += REPLACE_CHAR
    return _text_buf


def preprocess_text(text, max_sentence_length=127, min_sentence_length=10, DEBUG=False) -> list:


    #预处理规则：
    # 1、先简繁转换,再以‘。’对文本分段。
    # 2、段长度在min_sentence_length和max_sentence_length之间， 否则抛弃此段
    # 3、如段中没有汉字，则抛弃此段
    # 4、如段中最大连续汉字（含标点）长度小于min_sentence_length，如“句号 分段 还有89保留 句号  为此abc每个句号  后增加一空格”,则抛弃此段
    # 5  去除正文中的@和回复/转发中的用户名
    # 6  去除话题内容
    ss=''
    filter_punctuation = r'^[。，、：∶；?﹕︰﹔﹖﹑;！？！）〉›﹜』〖〗］》～｜]{1,}'
    if not isinstance(text, str):
        return text  # 2022-6-30
    
    #stop_terms = ['转发微博', '王一博','神韵','报导','澳洲','酪梨','萤幕','著','温哥华','瑜珈','甘迺迪','纽芬兰','多伦多','悉尼','萤光幕','麻糬','道家功法','渥太华','墨尔本','安大略','磅士卑','福克兰','加州','政法大学','签证','魁北克','四六','澳元','加兹尼','堪察加','蕃茄','本台消息','纽约','纽西兰','奥克兰','酮','镁铝','法大','奥勒冈','茼蒿','酚','二零','零二','零四','x ','六四分','修炼','指南指南','应援','汉子婊','谢谢谢谢','加拿大','神迹','快乐警察','不开心逮捕','音韵','这么','这个','这','辽宁','裸专','紫:','籽','隔空安静','师父']
    stop_terms = []
    txt = text.replace('\n', '')
    txt = txt.replace(',', '，')
    txt = txt.replace('\t', '')
    txt = txt.replace('祂', '他')
    txt = txt.replace('&lt', '<')
    txt = txt.replace('&gt', '>')
    txt = txt.replace('&amp', '&')
    txt = txt.replace('&quot', '"')
    txt = txt.replace('。”', '”。')  # 后引号”换到句号前面
    txt = txt.replace('。', '。〾')  # split('。')会吃掉‘。’， 故将‘。’换成‘。〾’以保留之 # 2022-6-12
    rawtext_list2 = txt.split('〾')  # 1、以‘〾’对文本分段。以句号结尾或连续两个句话时，会产生空！
    rawtext_list2 = txt.split('少时诵诗书所')

    cleantext_list = []
    rawtext_list=[]

    for sent in rawtext_list2:
        if len(re.findall(r"\S",sent))==0:
            continue  # 跳过
        else:
            rawtext_list.append(sent)
        txt = sent.replace('<url>','[url]')
        txt = txt.replace('<email>','[email]')
        txt = w3lib.html.remove_tags(txt)
        txt = zhconv.convert(txt, 'zh-hans')  # 简繁转换 
        URL_REGEX = re.compile(r'(?i)http[s]?://(?:[a-zA-Z]|[0-9]|[#$%*-;=?&@~.&+]|[!*,])+',
                    re.IGNORECASE)
        txt = re.sub(URL_REGEX, "[url]", txt)
        EMAIL_REGEX = re.compile(r"[-a-z0-9_.]+@(?:[-a-z0-9]+\.)+[a-z]{2,6}", re.IGNORECASE)
        txt = re.sub(EMAIL_REGEX, "[email]", txt)
        txt = txt.replace('.', '·')
        txt = txt.replace('‧', '')
        txt = txt.replace('•', '')
        #txt=re.sub(r'转发微博.*','',txt) # 
        #txt=re.sub(r'[\u0800-\u4e00]','',txt)# remove japanese
        txt=re.sub(r'[a-zA-Z\s]{8,}','',txt)#remove english
        #txt=re.sub(r'[uac00-ud7a3]','',txt)#remove korean
        txt = txt.replace('〖','【')
        txt = txt.replace('〗','】')
        txt = txt.replace('$','')
        txt = txt.replace('＂','')     
        txt = txt.replace('〉','')     
        txt = txt.replace('﹕','')     
        txt = txt.replace('）','')     
        txt = txt.replace('"','')     
        txt = txt.replace('##','')

        txt = txt.replace('～～','')     
        txt = txt.replace('〝','')     
        txt = txt.replace('〞','')
        txt = txt.replace('000','')
        #txt = txt.replace('RT@','')
        txt = re.sub(r'@[A-Za-z0-9_@-]{1,}','',txt)
        txt = re.sub(r'^RT','',txt)
        txt = re.sub(r'[﹕•:. （〉〈]','',txt)
        #txt = re.sub(r'^//','',txt)
        #txt = re.sub(r'^/','',txt)
        txt = re.sub(r'^[/﹕•:. （〉〈?!？！，…、～!：*]{1,}','',txt)
        for s in stop_terms:
            txt = txt.replace(s, '')
        txt = re.sub(r"(回复)?(//)?\s*@\S*?\s{1,10}(:|：| |$)", " ", txt)  # 去除正文中的@和回复/转发中的用户名
        txt = re.sub(r"#\S{1,20}#", "", txt)  # 去除话题内容
        if txt == '' or txt == ' ':
            cleantext_list.append(ss)
            continue  # 跳过
           
        words = filter_illegal_char(txt, REPLACE_CHAR='')
        punc = filter_punctuation
        words = re.sub(punc, '', words)
        if len(words) >= max_sentence_length:
            words = words[:max_sentence_length - 1]
        if len(words) < min_sentence_length:
            if DEBUG:
                print('舍弃段(规则2):', words)
            cleantext_list.append(ss)
            continue  # 2、段长度在min_sentence_length和max_sentence_length之间， 否则抛弃此段
            
        wordne = re.sub(r"\[\S+?\]", "", words)
        if len(wordne) < 6:
            if DEBUG:
                print('舍弃段(规则3_3):', words)
            cleantext_list.append(ss)
            continue  # 3、如段中unique汉字<5，则抛弃此段    
            
            
        if len(set(re.sub(r'[^\u4e00-\u9fa5]', '', words))) < 5:
            if DEBUG:
                print('舍弃段(规则3):', words)
            cleantext_list.append(ss)
            continue  # 3、如段中unique汉字<5，则抛弃此段
        if len((re.sub(r'[^\u4e00-\u9fa5]', '', words)))/len(set(re.sub(r'[^\u4e00-\u9fa5]', '', words))) > 4:
            if DEBUG:
                print('舍弃段(规则3_2):', words)
                '''
            words=re.sub('\[','',words)
            words=re.sub('\]','',words)
            #print(line3)
            pret=re.findall(r"(.{2,}).*\1",words)
            #print(pret)
            for i in pret:
                words=re.sub(str(i)+str(i),"",words)
            '''
            
            wordb=list(set(list(words)))
            worda=list(words)
            wordb.sort(key=worda.index)
            words=''.join(wordb)
            #cleantext_list.append(ss)
            #continue  # 3、如段中unique汉字<5，则抛弃此段
                
            
        splitte_words = re.split(r'[A-Za-z]', words)
        max_len = 0
        for sub_words in splitte_words:
            max_len = max(max_len, len(sub_words))
            
        if max_len < min_sentence_length:
            if DEBUG:
                print('舍弃段(规则4):', words)
                cleantext_list.append(ss)
                continue  # 4、如段中最大连续汉字（含标点）长度小于min_sentence_length，则抛弃此段
        # words is OK!
        words = re.sub('^ +| +$', '', words)  # 去掉首尾的空格
        cleantext_list.append(words)

    return cleantext_list


if __name__ == '__main__':

    s = "p 2 p 的就是习包子共产党精心策划的专门用来掠夺人民、抢劫人民、收割人民的一场政治运动#P2P#中共#诈骗"

    ss = filter_illegal_char(s)

    sss = preprocess_text(ss)

    pass