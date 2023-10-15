import re
s = "123655您好，欢迎来到我的博客：https://blog.csdn.net/weixin_44799217,,,###,,,我的邮箱是：535646343@qq.com. Today is 2021/12/21. It is a wonderful DAY!"
#  只匹配单一数字
rec = re.match(r'[0-9]{6}', s)   #  pattern匹配规则是从字符串开头数字匹配6次
print(rec)
print(rec.group())    # 打印出匹配到的值
ret = re.sub(r'[0-9]', "h", s) #将字符串s中的数字全部换成h
print(ret)

