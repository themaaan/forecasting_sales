import data_preprocessing as pre_processing
import test as t
import model as m
import data_post_processing as post_processing

def main():

    # -----------------------------------------
    # ------  Data preprocessing --------
    # -----------------------------------------
    pre_processing.main()


    # -----------------------------------------
    # ---------  Test for deploying -----------
    # -----------------------------------------
    t.main()
    

    # -----------------------------------------
    # -----------  Modelling  ---------------
    # -----------------------------------------
    m.main()

    # -----------------------------------------
    # ---------  Post-processing  -----------
    # -----------------------------------------
    post_processing.main()


if __name__ == "__main__":
    main()