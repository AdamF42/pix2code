import os

EVAL_SET = "../datasets/web/eval_set_originale/"
CODE_SET = "code/"


def compare(eval_el, code_el):
    correct = 0
    error = 0
    for i in range(0, min(len(eval_el), len(code_el))):
        if eval_el[i] == code_el[i]:
            correct += 1
        else:
            error += 1
    tot = correct + error
    return correct, error, tot


def check_existing_el(code_list, eval_list):
    result = all(elem in eval_list for elem in code_list)
    if result:
        print("Yes, eval_list contains all elements in code_list")
    else:
        print("No, eval_list does not contains all elements in code_list")


def print_accuracy(len_difference, tot_correct, tot_error, tot_tot):
    print("CORRETTI: ", tot_correct)
    print("ERRATI: ", tot_error + len_difference)
    print("TOTALI: ", tot_tot + len_difference)
    tot_correct_percentuale = (tot_correct / (tot_tot + len_difference)) * 100
    tot_error_percentuale = ((tot_error + len_difference) / (tot_tot + len_difference)) * 100
    print("PERCENTUALE CORRETTI: ", tot_correct_percentuale)
    print("PERCENTUALE ERRATI: ", tot_error_percentuale)
    assert round(tot_correct_percentuale, 2) + round(tot_error_percentuale, 2) == 100.0


def create_code_dict():
    res_code = {}  # chiave nome immagine, valore lista di stringhe (righe)
    count_code = 0
    code_list = []
    for filename in os.listdir(CODE_SET):
        if filename.endswith(".gui"):
            count_code += 1
            code_list.append(filename)
            with open(os.path.join(CODE_SET, f"{filename}"), 'r') as img:

                lines = []
                for el in img.readlines():
                    line = el.replace(" ", "  ") \
                        .replace(",", " ,") \
                        .replace("\n", " \n") \
                        .replace("{", " { ") \
                        .replace("}", " } ") \
                        .replace(",", " , ")
                    tokens = line.split(" ")
                    tokens = map(lambda x: " " if x == "" else x, tokens)
                    tokens = filter(lambda x: False if x == " " else True, tokens)
                    for token in tokens:
                        lines.append(token)

            res_code[filename] = lines
    assert (count_code == 250)
    return code_list, res_code


def create_eval_dict():
    res_eval = {}  # chiave nome immagine, valore lista di stringhe (righe)
    count_eval = 0
    eval_list = []
    for filename in os.listdir(EVAL_SET):
        if filename.endswith(".gui"):
            count_eval += 1
            eval_list.append(filename)
            with open(os.path.join(EVAL_SET, f"{filename}"), 'r') as img:

                lines = []
                for el in img.readlines():
                    line = el.replace(" ", "  ") \
                        .replace(",", " ,") \
                        .replace("\n", " \n")
                    tokens = line.split(" ")
                    tokens = map(lambda x: " " if x == "" else x, tokens)
                    tokens = filter(lambda x: False if x == " " else True, tokens)
                    for token in tokens:
                        lines.append(token)

            res_eval[filename] = lines
    assert (count_eval == 250)
    return eval_list, res_eval


def main():
    tot_correct = 0
    tot_error = 0
    tot_tot = 0
    len_difference = 0

    eval_list, res_eval = create_eval_dict()

    code_list, res_code = create_code_dict()

    check_existing_el(code_list, eval_list)

    for key in res_eval:
        if len(res_code[key]) != len(res_eval[key]):
            # se ho lunghezze diverse conto come errore la loro differenza
            len_difference += abs(len(res_code[key]) - len(res_eval[key]))
        corr, err, tot = compare(res_eval[key], res_code[key])
        tot_correct += corr
        tot_error += err
        tot_tot += tot

    print_accuracy(len_difference, tot_correct, tot_error, tot_tot)


if __name__ == "__main__":
    main()
