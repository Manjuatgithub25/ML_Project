import sys
import os
from logger import logger

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    line_num = exc_tb.tb_lineno
    error_message = "Error occured in python fille {0} at line no.{1}: error {2}".format(filename, line_num, str(error))

    return error_message

class CustomeException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        

    def __str__(self):
        return self.error_message



# if __name__ == '__main__':
#     try:
#         a = 1/0
#     except Exception as e:
#         logger.info(CustomeException(e, sys))
#         raise CustomeException(e, sys)
    
