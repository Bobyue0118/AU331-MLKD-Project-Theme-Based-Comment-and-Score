# -*- coding: utf-8 -*-
import requests
import re
import pandas as pd
import time
import csv             
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from bs4 import BeautifulSoup
import urllib

url = 'https://movie.douban.com/subject/{id}/comments?start={page}&limit=20&sort=new_score&status=P&percent_type={quality}'

# 读入需要爬取的电影文件（电影名+网址填充数字）
data = pd.read_csv('movie_to_crawl.csv')
movieId = list(data['number'])
movieName = list(data['film'])

commentType = ['h', 'm', 'l']  # 分别对应好评中评差评

def spider():
    s = requests.Session()
    # 登录网址
    loginUrl = 'https://accounts.douban.com/j/mobile/login/basic'
    # 登录信息
    formData = {
    "name": "18616550848",
    "password": "long19990227",
    "remember": 'False'
    }
    
    # 浏览器中网页的headers
    headers = {'User-Agent': "Mozilla/5.0", 'Referer': "https://accounts.douban.com/passport/login"}
    
    #登陆
    try:
        r = s.post(loginUrl, data=formData, headers=headers)
        r.raise_for_status()
    except:
        print("登录失败")

    page = r.text
    print(r.text)

    # 获取验证码
    soup = BeautifulSoup(page, "html.parser")
    captcha = soup.find('img', id='captcha_image')

    # 通过验证
    if captcha:
        captcha_url = captcha['src']
        reCaptchaID = r'<input type="hidden" name="captcha-id" value="(.*?)"/'
        captchaID = re.findall(reCaptchaID, page)
        #将网页中的验证码图片保存为captcha.jpg文件
        urllib.request.urlretrieve(captcha_url, "captcha.jpg")
        lena = mpimg.imread('captcha.jpg') # 读取和代码处于同一目录下的 lena.png
        plt.imshow(lena)  # 显示图片
        plt.axis('off')  # 不显示坐标轴
        plt.show()
        #请求输入验证码
        captcha_text = input('please input the captcha:')
        formData['captcha-solution'] = captcha_text
        formData['captcha-id'] = captchaID
        #输入完成后再次尝试登陆
        r = s.post(loginUrl, data=formData, headers=headers)
        page = r.text

    # 爬取数据
    for i in range(len(movieId)):
        print(movieName[i])
        movieNamelist = [movieName[i]]*20
        f = open('{}.csv'.format(movieName[i]), 'a+', newline='', encoding='utf-8-sig')
        w = csv.writer(f)
        for j in range(3):
            for page in range(0, 500, 20):  # 一种类型的影评从0-25页，每页20条
                url_tmp = url.format(id=movieId[i], page=page, quality=commentType[j])
                try:
                    html = s.get(url_tmp, headers=headers)
                    html.raise_for_status()
                except:
                    print("失败")
                # 爬取分数与评论
                scoreList = getScore(html)
                commentList = getComment(html)
                w.writerows(zip(commentList, scoreList, movieNamelist))
                time.sleep(2)  # 一页停2s
            time.sleep(2)
        f.close()
        time.sleep(5)  # 一部电影停5s


def getComment(html):
    commentList = re.findall('<span class="short">(.*)</span>', html.text)
    print(commentList)
    return commentList

def getScore(html):
    scoreList = re.findall('<span class="allstar(.*)0 rating"', html.text)
    # scoreList = []
    # for item in is_scoreList:
    #     if item.count('rating') == 0: #如果没有评分
    #         scoreList.append(0)
    #     else:
    #         scoreList.append(int(re.findall('<span class="allstar(.*)0 rating"', item)[0]))
    print(scoreList)
    return scoreList


if __name__ == "__main__" :
    spider()


