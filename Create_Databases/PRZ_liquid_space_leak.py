from Create_Databases.PRZ_space_leak_function import *
import MySQLdb

host = "localhost"  # 数据库地址
username = "root"  # 数据库用户名
passwd = "1111"  # 数据库密码
database = "mysql"  # 数据库类型

if __name__ == "__main__":
    # 获取获取本地路径
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    father_path = grandfather_path + 'Source Data\\Original Data'

    # 获取PRZ liquid space leak数据路径
    PRZ_liquid_path = father_path + '\\PRZ liquid space leak'
    keywords = ['PRZ liquid space leak']

    PRZ_liquid_data_list = []  # PRZ liquid space leak数据汇总
    PRZ_liquid_data_dict = dict()  # PRZ liquid space leak按文件给数据分组

    # 按文件读取数据
    temp_list, temp_dict = read_data(PRZ_liquid_path, keywords, 0, 50, 1001, 25)
    PRZ_liquid_data_list.append(temp_list)
    PRZ_liquid_data_dict.update(temp_dict)

    # 汇总RPZ liquid space leak数据
    PRZ_liquid = data_summary(PRZ_liquid_data_list)

    # 连接数据库
    db = MySQLdb.connect(host=host, user=username, passwd=passwd, db=database, charset='utf8')
    cursor = db.cursor()

    # 创建PRZ liquid space leak数据库
    cursor.execute("CREATE DATABASE IF NOT EXISTS PRZ_liquid_space_leak")
    cursor.execute("USE PRZ_liquid_space_leak")

    # 若表存在，则卸载该表
    temp_string_head = "DROP TABLE IF EXISTS PRZ_liquid_space_leak, "

    temp_string_body = drop_table(PRZ_liquid_path, keywords, 0, 50, 1001, 25)

    temp_string = temp_string_head + str(temp_string_body)
    cursor.execute(temp_string)

    # 为PRZ liquid space leak每个文件单独创建数据表
    create_single_table(PRZ_liquid_path, keywords, 0, 50, 1001, 25, db, PRZ_liquid_data_dict)

    # 创建PRZ liquid space leak总数据表
    create_summary_table(PRZ_liquid, db)
    print("==============================================================================================")
    print("已成功读取PRZ_liquid_space_leak数据")

    db.close()
