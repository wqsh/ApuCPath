
class CSystemInt:
    __UINT8_MIN = 0
    __UINT8_MAX = 255

    __UINT16_MIN = 0
    __UINT16_MAX = 65535

    __UINT32_MIN = 0
    __UINT32_MAX = 4294967295

    __UINT64_MIN = 0
    __UINT64_MAX = 18446744073709551615

    __INT8_MIN = -128
    __INT8_MAX = 127

    __INT16_MIN = -32768
    __INT16_MAX = 32767

    __INT32_MIN = -2147483648
    __INT32_MAX = 2147483647

    __INT64_MIN = -9223372036854775808
    __INT64_MAX = 9223372036854775807

    @staticmethod
    def INT_MIN():
        return CSystemInt.__INT32_MIN

    @staticmethod
    def INT_MAX():
        return CSystemInt.__INT32_MAX

    @staticmethod
    def UINT_MIN():
        return CSystemInt.__UINT32_MIN

    @staticmethod
    def UINT_MAX():
        return CSystemInt.__UINT32_MAX


INT_MIN = CSystemInt.INT_MIN()
INT_MAX = CSystemInt.INT_MAX()

UINT_MIN = CSystemInt.UINT_MIN()
UINT_MAX = CSystemInt.UINT_MAX()
