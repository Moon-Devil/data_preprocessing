from fault_diagnosis_function import *


def data_function(accident_name) -> object:
    x_data, _, _, _ = dataSet(accident_name)
    x_power = x_data[:301, 0]
    x_power_e = x_data[:301, 1]

    return x_power, x_power_e


def show_data():
    clear_file("A_power_h")
    clear_file("A_power_e")

    power_h, power_e = data_function("Ada@PowerR")
    write_to_text("A_power_h", power_h.tolist(), "a+")
    write_to_text("A_power_e", power_e.tolist(), "a+")

    power_h, power_e = data_function("Ada@PRLL")
    write_to_text("A_power_h", power_h.tolist(), "a+")
    write_to_text("A_power_e", power_e.tolist(), "a+")

    power_h, power_e = data_function("Ada@PRSL")
    write_to_text("A_power_h", power_h.tolist(), "a+")
    write_to_text("A_power_e", power_e.tolist(), "a+")

    power_h, power_e = data_function("Ada@CL_LOCA")
    write_to_text("A_power_h", power_h.tolist(), "a+")
    write_to_text("A_power_e", power_e.tolist(), "a+")

    power_h, power_e = data_function("Ada@HL_LOCA")
    write_to_text("A_power_h", power_h.tolist(), "a+")
    write_to_text("A_power_e", power_e.tolist(), "a+")

    power_h, power_e = data_function("Ada@SG2L")
    write_to_text("A_power_h", power_h.tolist(), "a+")
    write_to_text("A_power_e", power_e.tolist(), "a+")

    power_h, power_e = data_function("Ada@SGTR")
    write_to_text("A_power_h", power_h.tolist(), "a+")
    write_to_text("A_power_e", power_e.tolist(), "a+")


show_data()
