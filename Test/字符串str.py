s = "deef"
print(s.capitalize())   #将字符串的首字母转换为大写
br = "哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈"
print(br.count("哈哈"))    #统计字符串出现的次数
bt = "sdfhjsdfh"
print(bt.endswith("fh"))  #统计字符串是否以fh结尾
bc = "adfsdfdsd"
print(bc.index("f")) #查询字符或字符串在字符串中第一次出现的位置,如果没有则抛出异常
bc = "adfsdfdsd"
print(bc.rindex("f"))  #从右往左查询字符或字符串在字符串中第一次出现的位置，如果没有则抛出异常
# find() 查询字符或字符串在字符串中第一次出现的位置，如果没有则会返回-1
# islower() 判断字母是否全为小写
# lower() 将字符全部转换为小写
# split() 将字符串按照特定的格式进行分割，返回值是一个列表
s = "hello world"
print(s.split("o"))
# join() 按照特定的符号将一个可迭代对象拼接成字符串
s = ["a", "b", "c"]
print("se".join(s))
m = "abc"
print("se".join(m))