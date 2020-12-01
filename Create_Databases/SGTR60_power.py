from Create_Databases.SGTR_power_function import *
import MySQLdb

host = "localhost"  # 数据库地址
username = "root"  # 数据库用户名
passwd = "1111"  # 数据库密码
database = "mysql"  # 数据库类型

if __name__ == "__main__":
    # 获取获取本地路径
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    father_path = grandfather_path + 'Source Data\\Original Data'

    # 获取SGTR60 power数据路径
    SGTR60_power_path = father_path + '\\SGTR60 power'
    keywords = ['SGTR60_']

    SGTR60_power_data_list = []  # SGTR60 power数据汇总
    SGTR60_power_data_dict = dict()  # SGTR60 power按文件给数据分组

    # 按文件读取数据
    temp_list, temp_dict = read_data(SGTR60_power_path, keywords, 0, 5, 101, 1)
    SGTR60_power_data_list.append(temp_list)
    SGTR60_power_data_dict.update(temp_dict)

    # 汇总SGTR60 power数据
    SGTR60_power = data_summary(SGTR60_power_data_list)

    # 连接数据库
    db = MySQLdb.connect(host=host, user=username, passwd=passwd, db=database, charset='utf8')
    cursor = db.cursor()

    # 创建SGTR60 power数据库
    cursor.execute("CREATE DATABASE IF NOT EXISTS SGTR60_power")
    cursor.execute("USE SGTR60_power")

    # 若表存在，则卸载该表
    temp_string_head = "DROP TABLE IF EXISTS SGTR60_power, "

    temp_string_body = drop_table(SGTR60_power_path, keywords, 0, 5, 101, 1)

    temp_string = temp_string_head + str(temp_string_body)
    cursor.execute(temp_string)

    # 为SGTR60 power每个文件单独创建数据表
    create_single_table(SGTR60_power_path, keywords, 0, 5, 101, 1, db, SGTR60_power_data_dict)

    # 创建SGTR60 power总数据表
    create_summary_table(SGTR60_power, db)
    print("==============================================================================================")
    print("已成功读取SGTR60 power数据")

    db.close()
