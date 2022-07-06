import csv
import numpy as np
import DNN_data


def test():
    filename = 'data2csv/Italia_data.csv'
    date, data = DNN_data.load_2csvData(datafile=filename)
    print(date)
    print('\n')
    print(data)

    index = np.random.randint(5, size=2)
    print(index)

    data1samples, data2samples = DNN_data.randSample_existData(date, data, batchsize=2)
    print(data1samples)
    print(data2samples)


def test1():
    filename = 'data2csv/Italia_data.csv'
    date, data2I,  data2S= DNN_data.load_2csvData_cal_S(datafile=filename,total_population=500000)
    print(date)
    print('\n')
    print(data2I)
    print(data2S)

    index = np.random.randint(5, size=2)
    print(index)

    data1samples, data2samples = DNN_data.randSample_existData(date, data2I, batchsize=2)
    print(data1samples)
    print(data2samples)


def test2():
    filename = 'data2csv/Italia_data.csv'
    date, data1, data2 = DNN_data.load_3csvData(filename)
    print(date)
    print('\n')
    print(data1)
    print('\n')
    print(data2)

    index = np.random.randint(5, size=2)
    print(index)

    date_samples, data1samples, data2samples = DNN_data.randSample_3existData(date, data1, data2, batchsize=2)
    print(date_samples)
    print(data1samples)
    print(data2samples)


if __name__ == "__main__":
    # test()
    test1()
    # test2()