from Create_Databases.power_decreasing_function import *
import MySQLdb

host = 'localhost'  # 数据库地址
username = 'root'  # 数据库用户名
passwd = '1111'  # 数据库密码
database = 'mysql'  # 数据库类型

if __name__ == "__main__":
    # 获取获取本地路径
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    father_path = grandfather_path + 'Source Data\\Original Data'

    # 获取power decreasing数据路径
    power_decreasing_path = father_path + '\\power decreasing'
    keywords = ['target200 rate', 'target220 rate', 'target250 rate']

    power_decreasing_data_list = []  # power decreasing数据汇总
    power_decreasing_data_dict = dict()  # power decreasing按rate数据分组

    # 按文件读取数据
    temp_list, temp_dict = read_data(power_decreasing_path, keywords, 0, 30, 36, 1)
    power_decreasing_data_list.append(temp_list)
    power_decreasing_data_dict.update(temp_dict)

    temp_list, temp_dict = read_data(power_decreasing_path, keywords, 1, 22, 29, 1)
    power_decreasing_data_list.append(temp_list)
    power_decreasing_data_dict.update(temp_dict)

    temp_list, temp_dict = read_data(power_decreasing_path, keywords, 2, 5, 21, 0.5)
    power_decreasing_data_list.append(temp_list)
    power_decreasing_data_dict.update(temp_dict)

    # 汇总power decreasing数据
    power_decreasing = data_summary(power_decreasing_data_list)

    # 连接数据库
    db = MySQLdb.connect(host=host, user=username, passwd=passwd, db=database, charset='utf8')
    cursor = db.cursor()

    # 创建power_decreasing数据库
    cursor.execute("CREATE DATABASE IF NOT EXISTS power_decreasing")
    cursor.execute("USE power_decreasing")

    # 若表存在，则卸载该表
    temp_string_head: str = "DROP TABLE IF EXISTS power_decreasing, "

    temp_string_body_1 = drop_table(power_decreasing_path, keywords, 0, 30, 36, 1)
    temp_string_body_2 = drop_table(power_decreasing_path, keywords, 1, 22, 29, 1)
    temp_string_body_3 = drop_table(power_decreasing_path, keywords, 2, 5, 21, 0.5)
    temp_string_list = [temp_string_body_1, ", ", temp_string_body_2, ", ", temp_string_body_3]
    temp_string_body = "".join(temp_string_list)

    temp_string = temp_string_head + temp_string_body
    cursor.execute(temp_string)

    # 为power_decreasing每个文件单独创建数据表
    create_single_table(power_decreasing_path, keywords, 0, 30, 36, 1, db, power_decreasing_data_dict)
    create_single_table(power_decreasing_path, keywords, 1, 22, 29, 1, db, power_decreasing_data_dict)
    create_single_table(power_decreasing_path, keywords, 2, 5, 21, 0.5, db, power_decreasing_data_dict)

    # 创建power_decreasing总数据表
    create_summary_table(power_decreasing, db)
    print("==============================================================================================")
    print("已成功读取power_decreasing数据")

    db.close()
