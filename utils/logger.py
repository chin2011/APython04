import logging
import os
from datetime import datetime


def setup_logger(log_dir="log", name="train"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)


def log_dm01():
    logger = setup_logger('../log', 'demo')
    logger.info(f"当前时间: {datetime.now()}")
    logger.info("开始计算...")
    try:
        a = 10 / 0
    except Exception as e:
        logger.error(f"错误信息: {e}")
    else:
        logger.info("计算成功")
    finally:
        logger.info("计算结束")
        

# 2.生成train_年月日.Log日志文件。
def log_dm02():
    logger = setup_logger(log_dir=f"../log",
                          name=f"train_{datetime.now().strftime('%Y-%m-%d')}")
    logger.info("开始计算...")
    a = 10 / 2
    logger.info(f"计算结果: {a}")
    logger.info("计算结束")
    
    
    
if __name__ == '__main__':
    # log_dm01()
    log_dm02()
    
    