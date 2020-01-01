import jieba
import pandas as pd
import re
import os
import glob
import csv

aa = 'csv'
a = [i for i in glob.glob('*.{}'.format(aa))]  # 加载所有后缀为csv的文件。
if '20_movies.csv' in a:
    a.delete('20_movies.csv')

with open('20_movies.csv', 'a+',  newline='', encoding='utf-8-sig') as f:
    for i in a:
        data = pd.read_csv(i, header=None, encoding='utf-8')
        # comment = list(data.iloc[:, 0])
        # score = list(data.iloc[:, 1])
        # movie = list(data.iloc[:, 2])
        csv_write = csv.writer(f)
        df = list(data.iloc[0])
        for j in range(len(data)):
            csv_write.writerow(list(data.iloc[j]))

name = '20_movies.csv'
generate_name = 'processed_data_20_movies.txt'
rule = re.compile(u'[^\u4E00-\u9FA5]')

if __name__ == '__main__':
    #读入影评+评分数据
    data = pd.read_csv(name, encoding='utf-8', header=None)
    comment = list(data.iloc[:, 0])
    score = list(data.iloc[:, 1])
    movie = list(data.iloc[:, 2])
    # with open(name, 'rb') as f:
    #     text = f.read()

    with open(generate_name, 'w', encoding='utf-8') as f:
        # for i in range(len(data)):
        #     if type(comment[i]) != float:
        #         tmp_com = comment[i]
        for tmp_com, tmp_sco, tmp_mov in zip(comment, score, movie):
            if(type(tmp_com)==float):
                continue
            if len(tmp_com) < 6:
                continue
            if bool(re.search('[a-z]', tmp_com)):
                continue
            tmp_com = tmp_com.strip()
            tmp_com = tmp_com.replace('\r\n', '')
            tmp_com = tmp_com.replace('\r', '')
            tmp_com = tmp_com.replace('\n', '')
            tmp_com = ''.join(tmp_com.split())

            punctuation = """！!？?｡,，＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～~｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
            re_punctuation = "[{}]+".format(punctuation)
            new_com = re.sub(re_punctuation, "", tmp_com)
            punctuation = [',', '，', '.', '。', '！']
            for pun in punctuation:
                new_com = new_com.strip(pun)
            ret = rule.findall(new_com)
            if(bool(rule.findall(new_com))):
                continue

            if type(tmp_com) == float:
                continue
            seg_list = jieba.cut(tmp_com, cut_all=False)
            f.write(' '.join(seg_list))
            f.write(' </d> ')
            f.write(tmp_mov)
            f.write(' ')
            tmp_sco = int(tmp_sco)
            if tmp_sco == 1:
                f.write('一星级')
            elif tmp_sco == 2:
                f.write('二星级')
            elif tmp_sco == 3:
                f.write('三星级')
            elif tmp_sco == 4:
                f.write('四星级')
            else:
                f.write('五星级')
            f.write('\n')
# if __name__ == '__main__':
#     # 读入影评+评分数据
#
#     data = pd.read_csv(name, encoding='utf-8')
#     comment = list(data.iloc[:, 0])
#     score = list(data.iloc[:, 1])
#     movie = list(data.iloc[:, 2])
#     # with open(name, 'rb') as f:
#     #     text = f.read()
#
#     with open(generate_name, 'w', encoding='utf-8') as f:
#         # for i in range(len(data)):
#         #     if type(comment[i]) != float:
#         #         tmp_com = comment[i]
#         for tmp_com, tmp_sco, tmp_mov in zip(comment, score, movie):
#             if(type(tmp_com)==float):
#                 continue
#             if len(tmp_com) < 6:
#                 continue
#             if bool(re.search('[a-z]', tmp_com)):
#                 continue
#             tmp_com = tmp_com.strip()
#             tmp_com = tmp_com.replace('\r\n', '')
#             tmp_com = tmp_com.replace('\r', '')
#             tmp_com = tmp_com.replace('\n', '')
#             tmp_com = ''.join(tmp_com.split())
#
#             punctuation = """！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
#             re_punctuation = "[{}]+".format(punctuation)
#             new_com = re.sub(re_punctuation, "", tmp_com)
#             ret = rule.match(new_com)
#             if(ret != None):
#                 continue
#
#             if type(tmp_com) == float:
#                 continue
#             seg_list = jieba.cut(tmp_com, cut_all=False)
#             f.write(' '.join(seg_list))
#             f.write(' </d> ')
#             f.write(tmp_mov)
#             f.write(' ')
#             tmp_sco = int(tmp_sco)
#             if tmp_sco == 1:
#                 f.write('一星级')
#             elif tmp_sco == 2:
#                 f.write('二星级')
#             elif tmp_sco == 3:
#                 f.write('三星级')
#             elif tmp_sco == 4:
#                 f.write('四星级')
#             else:
#                 f.write('五星级')
#             f.write('\n')
#
