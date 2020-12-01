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

    # 获取SG_2nd_side_leak数据路径
    SG_2nd_side_leak_path = father_path + '\\SG 2nd side leak'
    keywords = ['SG 2nd side leak']

    SG_2nd_side_leak_data_list = []  # SG 2nd side leak数据汇总
    SG_2nd_side_leak_data_dict = dict()  # SG 2nd side leak按文件给数据分组

    # 按文件读取数据
    temp_list, temp_dict = read_data(SG_2nd_side_leak_path, keywords, 0, 50, 1001, 25)
    SG_2nd_side_leak_data_list.append(temp_list)
    SG_2nd_side_leak_data_dict.update(temp_dict)

    # 汇总SG 2nd side leak数据
    SG_2nd_side_leak = data_summary(SG_2nd_side_leak_data_list)

    # 连接数据库
    db = MySQLdb.connect(host=host, user=username, passwd=passwd, db=database, charset='utf8')
    cursor = db.cursor()

    # 创建SG 2nd side leak数据库
    cursor.execute("CREATE DATABASE IF NOT EXISTS SG_2nd_side_leak")
    cursor.execute("USE SG_2nd_side_leak")

    # 若表存在，则卸载该表
    temp_string_head = "DROP TABLE IF EXISTS SG_2nd_side_leak, "

    temp_string_body = drop_table(SG_2nd_side_leak_path, keywords, 0, 50, 1001, 25)

    temp_string = temp_string_head + str(temp_string_body)
    cursor.execute(temp_string)

    # 为SG 2nd side leak每个文件单独创建数据表
    create_single_table(SG_2nd_side_leak_path, keywords, 0, 50, 1001, 25, db, SG_2nd_side_leak_data_dict)

    # 创建SG 2nd side leak总数据表
    create_summary_table(SG_2nd_side_leak, db)
    print("==============================================================================================")
    print("已成功读取SG 2nd side leak数据")

    db.close()
