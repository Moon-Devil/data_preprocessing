from Data_Pre_analysis.power_decreasing_pre_analysis_fuction import *

target200_rate30_array = data_from_database("power_decreasing", "target200_rate30", 200, 30)
target200_rate32_array = data_from_database("power_decreasing", "target200_rate32", 200, 32)
target200_rate34_array = data_from_database("power_decreasing", "target200_rate34", 200, 34)
target200_rate35_array = data_from_database("power_decreasing", "target200_rate35", 200, 35)

target220_rate22_array = data_from_database("power_decreasing", "target220_rate22", 220, 22)
target220_rate23_array = data_from_database("power_decreasing", "target220_rate23", 220, 23)
target220_rate24_array = data_from_database("power_decreasing", "target220_rate24", 220, 24)
target220_rate25_array = data_from_database("power_decreasing", "target220_rate25", 220, 25)
target220_rate26_array = data_from_database("power_decreasing", "target220_rate26", 220, 26)
target220_rate27_array = data_from_database("power_decreasing", "target220_rate27", 220, 27)
target220_rate28_array = data_from_database("power_decreasing", "target220_rate28", 220, 28)

target250_rate5_array = data_from_database("power_decreasing", "target250_rate5", 250, 5)
target250_rate7_5_array = data_from_database("power_decreasing", "target250_rate7_5", 250, 7.5)
target250_rate10_array = data_from_database("power_decreasing", "target250_rate10", 250, 10)
target250_rate12_array = data_from_database("power_decreasing", "target250_rate12", 250, 12)
target250_rate13_array = data_from_database("power_decreasing", "target250_rate13", 250, 13)
target250_rate14_array = data_from_database("power_decreasing", "target250_rate14", 250, 14)
target250_rate15_array = data_from_database("power_decreasing", "target250_rate15", 250, 15)
target250_rate16_array = data_from_database("power_decreasing", "target250_rate16", 250, 16)
target250_rate17_array = data_from_database("power_decreasing", "target250_rate17", 250, 17)
target250_rate18_array = data_from_database("power_decreasing", "target250_rate18", 250, 18)
target250_rate20_array = data_from_database("power_decreasing", "target250_rate20", 250, 20)

# power decreasing数据数组
power_decreasing = np.vstack([target200_rate30_array, target200_rate32_array, target200_rate34_array,
                              target200_rate35_array, target220_rate22_array, target220_rate23_array,
                              target220_rate24_array, target220_rate25_array, target220_rate26_array,
                              target220_rate27_array, target220_rate28_array, target250_rate5_array,
                              target250_rate7_5_array, target250_rate10_array, target250_rate12_array,
                              target250_rate13_array, target250_rate14_array, target250_rate15_array,
                              target250_rate16_array, target250_rate17_array, target250_rate18_array,
                              target250_rate20_array])

# power decreasing各文件的长度
power_decreasing_nodes = [len(target200_rate30_array), len(target200_rate32_array), len(target200_rate34_array),
                          len(target200_rate35_array), len(target220_rate22_array), len(target220_rate23_array),
                          len(target220_rate24_array), len(target220_rate25_array), len(target220_rate26_array),
                          len(target220_rate27_array), len(target220_rate28_array), len(target250_rate5_array),
                          len(target250_rate7_5_array), len(target250_rate10_array), len(target250_rate12_array),
                          len(target250_rate13_array), len(target250_rate14_array), len(target250_rate15_array),
                          len(target250_rate16_array), len(target250_rate17_array), len(target250_rate18_array),
                          len(target250_rate20_array)]
