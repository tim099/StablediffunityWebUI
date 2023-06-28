

def get_arg_val(args: dict, key: str, default_val):
    try:
        val = default_val
        if(key in args):
            if(type(val) is bool):
                val = args[key] == 'True'
            else:
                val = args[key]
        else:
            print("get_arg_val key:"+key+",not exist!!")
    except BaseException:
        print(f"get_arg_val key:"+key+",BaseException:{BaseException}")
    finally:
        return val;