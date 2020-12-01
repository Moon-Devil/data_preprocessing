from Create_Databases.RCS_LOCA_function import *
import MySQLdb

host = "localhost"  # 数据库地址
username = "root"  # 数据库用户名
passwd = "1111"  # 数据库密码
database = "mysql"  # 数据库类型

if __name__ == "__main__":
    # 获取获取本地路径
    grandfather_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../.."))  # 获取父目录路径
    father_path = grandfather_path + 'Source Data\\Original Data'

    # 获取RCS CL LOCA 2数据路径
    RCS_CL_LOCA_2_path = father_path + '\\RCS CL LOCA 2'
    keywords = ['RCS CL LOCA60_']

    RCS_CL_LOCA_2_data_list = []  # RCS_CL_LOCA_2数据汇总
    RCS_CL_LOCA_2_data_dict = dict()  # RCS_CL_LOCA_2按文件给数据分组

    # 按文件读取数据
    temp_list, temp_dict = read_data(RCS_CL_LOCA_2_path, keywords, 0, 1, 21, 1)
    RCS_CL_LOCA_2_data_list.append(temp_list)
    RCS_CL_LOCA_2_data_dict.update(temp_dict)

    # 汇总RCS CL LOCA 2数据
    RCS_CL_LOCA_2 = data_summary(RCS_CL_LOCA_2_data_list)
    # 连接数据库
    db = MySQLdb.connect(host=host, user=username, passwd=passwd, db=database, charset='utf8')
    cursor = db.cursor()

    # 创建RCS CL LOCA 2数据库
    cursor.execute("CREATE DATABASE IF NOT EXISTS RCS_CL_LOCA_2")
    cursor.execute("USE RCS_CL_LOCA_2")

    # 若表存在，则卸载该表
    temp_string_head = "DROP TABLE IF EXISTS RCS_CL_LOCA_2, "

    temp_string_body = drop_table(RCS_CL_LOCA_2_path, keywords, 0, 1, 21, 1)

    temp_string = temp_string_head + str(temp_string_body)
    cursor.execute(temp_string)

    # 为RCS CL LOCA 2每个文件单独创建数据表
    create_single_table(RCS_CL_LOCA_2_path, keywords, 0, 1, 21, 1, db, RCS_CL_LOCA_2_data_dict)

    # 创建RCS CL LOCA 2总数据表
    create_summary_table(RCS_CL_LOCA_2, db)
    print("==============================================================================================")
    print("已成功读取RCS CL LOCA 2数据")

    db.close()
