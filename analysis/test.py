import logging

def a():
    print("a")

    logging.basicConfig(filename="../results/run_2/a.log",
                        filemode='a',
                        format='%(asctime)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logging.info("abc")



if __name__ == '__main__':
    a()
