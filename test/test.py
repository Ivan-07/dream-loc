import xml.etree.cElementTree as ET

import torch


def load_and_split_reports(report_path, split_ratio='8:1:1'):
    """
    sort reports by 'report_timestamp', then split.
    """
    root = ET.parse(report_path).getroot()
    reports = list(root.iter('table'))
    reports = list(sorted(reports, key=lambda x: int(x[5].text)))
    n_train = int(int(split_ratio.split(':')[0]) / 10 * len(reports))
    n_val = int(int(split_ratio.split(':')[1]) / 10 * len(reports))
    train_reports = reports[:n_train]
    val_reports = reports[n_train: n_train + n_val]
    test_reports = reports[n_train + n_val:]
    print(f'split_ratio = {split_ratio}')
    print(f'train reports: {len(train_reports)}')
    print(f'val   reports: {len(val_reports)}')
    print(f'test  reports: {len(test_reports)}')
    return train_reports, val_reports, test_reports


if __name__ == '__main__':
    # s = ['java/org/apache/catalina/startup/Catalina.java', 'java/org/apache/tomcat/util/digester/Digester.java']
    # for idx in range(len(s)):
    #     s[idx] = s[idx].replace('/', '\\')
    #     print(s[idx])
    print(torch.version.cuda)
    print(torch.__version__)  # 注意是双下划线
