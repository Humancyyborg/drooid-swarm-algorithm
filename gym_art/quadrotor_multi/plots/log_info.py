import csv


def log_list_data(filename, data):
    # Writing to a CSV file
    with open(filename, mode='w', newline='') as file:
        # Prepare header
        header = []
        for key in data[0].keys():
            for i in range(len(data[0][key])):
                header.append(f"{key}_{i}")

        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()

        for entry in data:
            row = {}
            for key, array in entry.items():
                for i, value in enumerate(array):
                    row[f"{key}_{i}"] = value
            writer.writerow(row)


def log_data_v1(filename, data):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['X', 'Y', 'Z'])  # Header
        writer.writerows(data)


def log_init_info(filename, data):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
