import os
import MySQLdb

host = "localhost"  # 数据库地址
username = "root"  # 数据库用户名
passwd = "1111"  # 数据库密码
database = "mysql"  # 数据库类型

if __name__ == "__main__":
    os.system("python power_decreasing.py")
    os.system("python PRZ_liquid_space_leak.py")
    os.system("python PRZ_vapour_space_leak.py")
    os.system("python RCS_CL_LOCA_1.py")
    os.system("python RCS_CL_LOCA_2.py")
    os.system("python RCS_HL_LOCA_1.py")
    os.system("python RCS_HL_LOCA_2.py")
    os.system("python SG_2nd_side_leak.py")
    os.system("python SGTR60_power.py")
    os.system("python SGTR100_power.py")
    print("==============================================================================================")
    print("数据存储已完成！！！")

    # 统计数据库数据量
    data_counts = 0
    db = MySQLdb.connect(host=host, user=username, passwd=passwd, db=database, charset='utf8')
    cursor = db.cursor()
    sql = "USE power_decreasing"
    cursor.execute(sql)
    sql = "SElECT count(*) from all_data"
    cursor.execute(sql)
    power_decreasing = cursor.fetchone()

    cursor = db.cursor()
    sql = "USE PRZ_liquid_space_leak"
    cursor.execute(sql)
    sql = "SElECT count(*) from all_data"
    cursor.execute(sql)
    PRZ_liquid_space_leak = cursor.fetchone()

    cursor = db.cursor()
    sql = "USE PRZ_vapour_space_leak"
    cursor.execute(sql)
    sql = "SElECT count(*) from all_data"
    cursor.execute(sql)
    PRZ_vapour_space_leak = cursor.fetchone()

    cursor = db.cursor()
    sql = "USE RCS_CL_LOCA_1"
    cursor.execute(sql)
    sql = "SElECT count(*) from all_data"
    cursor.execute(sql)
    RCS_CL_LOCA_1 = cursor.fetchone()

    cursor = db.cursor()
    sql = "USE RCS_CL_LOCA_2"
    cursor.execute(sql)
    sql = "SElECT count(*) from all_data"
    cursor.execute(sql)
    RCS_CL_LOCA_2 = cursor.fetchone()

    cursor = db.cursor()
    sql = "USE RCS_HL_LOCA_1"
    cursor.execute(sql)
    sql = "SElECT count(*) from all_data"
    cursor.execute(sql)
    RCS_HL_LOCA_1 = cursor.fetchone()

    cursor = db.cursor()
    sql = "USE RCS_HL_LOCA_2"
    cursor.execute(sql)
    sql = "SElECT count(*) from all_data"
    cursor.execute(sql)
    RCS_HL_LOCA_2 = cursor.fetchone()

    cursor = db.cursor()
    sql = "USE SG_2nd_side_leak"
    cursor.execute(sql)
    sql = "SElECT count(*) from all_data"
    cursor.execute(sql)
    SG_2nd_side_leak = cursor.fetchone()

    cursor = db.cursor()
    sql = "USE SGTR60_power"
    cursor.execute(sql)
    sql = "SElECT count(*) from all_data"
    cursor.execute(sql)
    SGTR60_power = cursor.fetchone()

    cursor = db.cursor()
    sql = "USE SGTR_power"
    cursor.execute(sql)
    sql = "SElECT count(*) from all_data"
    cursor.execute(sql)
    SGTR_power = cursor.fetchone()
    data_counts = power_decreasing[0] + PRZ_liquid_space_leak[0] + PRZ_vapour_space_leak[0] + RCS_CL_LOCA_1[0] \
                  + RCS_CL_LOCA_2[0] + RCS_HL_LOCA_1[0] + RCS_HL_LOCA_2[0] + SG_2nd_side_leak[0] + SGTR60_power[0]\
                  + SGTR_power[0]

    print("==============================================================================================")
    print("各数据表的数据量")
    print("----------------------------------------------------------------------------------------------")
    print("power decreasing\t\t\t\t" + str(power_decreasing[0]))
    print("PRZ_liquid_space_leak\t\t\t" + str(PRZ_liquid_space_leak[0]))
    print("PRZ_vapour_space_leak\t\t\t" + str(PRZ_vapour_space_leak[0]))
    print("RCS_CL_LOCA_1\t\t\t\t\t" + str(RCS_CL_LOCA_1[0]))
    print("RCS_CL_LOCA_2\t\t\t\t\t" + str(RCS_CL_LOCA_2[0]))
    print("RCS_HL_LOCA_1\t\t\t\t\t" + str(RCS_HL_LOCA_1[0]))
    print("RCS_HL_LOCA_2\t\t\t\t\t" + str(RCS_HL_LOCA_2[0]))
    print("SG_2nd_side_leak\t\t\t\t" + str(SG_2nd_side_leak[0]))
    print("SGTR60_power\t\t\t\t\t" + str(SGTR60_power[0]))
    print("SGTR_power\t\t\t\t\t\t" + str(SGTR_power[0]))
    print("----------------------------------------------------------------------------------------------")
    print("数据总量\t\t\t\t\t\t\t" + str(data_counts))

    db.close()

