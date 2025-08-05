import os
import sys
import traceback

def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()
    file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    error_message = "Error occurred python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message


class CustomerException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = CustomerException.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys):
        try:
            _, _, exc_tb = sys.exc_info()
            if exc_tb is not None:
                file_name = exc_tb.tb_frame.f_code.co_filename
                line_number = exc_tb.tb_lineno
                return f"Error occurred in script: [{file_name}] at line number: [{line_number}] with message: [{str(error_message)}]"
            else:
                return f"Error: {str(error_message)} (no traceback available)"
        except Exception as internal_error:
            return f"Error while logging error: {internal_error}, original error: {error_message}"

    def __str__(self):
        return self.error_message

