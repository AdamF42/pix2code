import os

EVAL_SET = "eval_set/"
CODE_SET = "code/"


def confronta(eval_el, code_el):
    correct = 0
    error = 0
    if type(eval_el) is str and type(code_el) is str:
        if eval_el == code_el:
            correct += 1
        else:
            error += 1
    if type(eval_el) is str and type(code_el) is list:
        error += len(code_el)  # - 1
    if type(eval_el) is list and type(code_el) is str:
        error += len(eval_el)  # - 1
    if type(eval_el) is list and type(code_el) is list:
        for i in range(0, min(len(eval_el), len(code_el))):
            if eval_el[i] == code_el[i]:
                correct += 1
            else:
                error += 1
    tot = correct + error
    return correct, error, tot


def main():
    tot_correct = 0
    tot_error = 0
    tot_tot = 0
    bug = 0
    mini_bug = 0

    res_eval = {}  # chiave nome immagine, valore lista di stringhe (righe) 
    count = 0
    eval_list = []
    for filename in os.listdir(EVAL_SET):
        if filename.endswith(".gui"):
            count += 1
            eval_list.append(filename)
            with open(os.path.join(EVAL_SET, f"{filename}"), 'r') as img:
                # lines = img.readlines()

                lines = []
                for el in img.readlines():  # ci deve sempre essere uno spazio fra ogni token sennò non va
                    line = el.replace(" ", "  ").replace(",", " ,").replace("\n", " \n")
                    tokens = line.split(" ")
                    tokens = map(lambda x: " " if x == "" else x, tokens)
                    tokens = filter(lambda x: False if x == " " else True, tokens)
                    for token in tokens:
                        lines.append(token)

                # lines = [
                #     el.strip(' ').replace(', ', ',').replace('} ', '}').replace('{ ', '{').replace(' }', '}').replace(
                #         ' {', '{') for el in lines]
            res_eval[filename] = lines
    assert (count == 250)

    res_code = {}  # chiave nome immagine, valore lista di stringhe (righe)
    count0 = 0
    code_list = []
    for filename in os.listdir(CODE_SET):
        if filename.endswith(".gui"):
            count0 += 1
            code_list.append(filename)
            with open(os.path.join(CODE_SET, f"{filename}"), 'r') as img:
                # lines = img.readlines()

                lines = []
                for el in img.readlines():  # ci deve sempre essere uno spazio fra ogni token sennò non va
                    line = el.replace(" ", "  ")\
                        .replace(",", " ,")\
                        .replace("\n", " \n")\
                        .replace("{", " { ")\
                        .replace("}", " } ")\
                        .replace(",", " , ")
                    tokens = line.split(" ")
                    tokens = map(lambda x: " " if x == "" else x, tokens)
                    tokens = filter(lambda x: False if x == " " else True, tokens)
                    for token in tokens:
                        lines.append(token)

            res_code[filename] = lines
    assert (count0 == 250)

    result = all(elem in eval_list for elem in code_list)
    if result:
        print("Yes, eval_list contains all elements in code_list")
    else:
        print("No, eval_list does not contains all elements in code_list")

    for key in res_eval:
        e = res_eval[key]
        c = res_code[key]
        if len(c) != len(e):
            bug += abs(len(e) - len(c))  # lunghezze diverse conto come errore la loro differenza
            mini_bug += 1  # lunghezze diverse conto come errore solamente 1
        for i in range(min(len(e), len(c))):
            # leggendo i file per riga, nelle righe relative ai bottoni c'è sempre la virgola.
            # Esempio: btn-red, btn-green, btn-black.
            # Nelle righe senza bottone non ci sono mai virgole
            if "," in e[i]:
                e[i] = e[i].split(",")
            if "," in c[i]:
                c[i] = c[i].split(",")
            corr, err, tot = confronta(e[i], c[i])
            tot_correct += corr
            tot_error += err
            tot_tot += tot

    print(f"CON {bug} BUG")
    print("CORRETTI: ", tot_correct)
    print("ERRATI: ", tot_error + bug)
    print("TOTALI: ", tot_tot + bug)
    tot_correct_percentuale = (tot_correct / (tot_tot + bug)) * 100
    tot_error_percentuale = ((tot_error + bug) / (tot_tot + bug)) * 100
    print(tot_correct_percentuale)
    print(tot_error_percentuale)
    assert round(tot_correct_percentuale, 2) + round(tot_error_percentuale, 2) == 100.0

    # print("-" * 50)
    # print(f"CON {mini_bug} BUG")
    # print("CORRETTI: ", tot_correct)
    # print("ERRATI: ", tot_error + mini_bug)
    # print("TOTALI: ", tot_tot + mini_bug)
    # tot_correct_percentuale = (tot_correct / (tot_tot + mini_bug)) * 100
    # tot_error_percentuale = ((tot_error + mini_bug) / (tot_tot + mini_bug)) * 100
    # print(tot_correct_percentuale)
    # print(tot_error_percentuale)
    # assert round(tot_correct_percentuale, 2) + round(tot_error_percentuale, 2) == 100.0
    #
    # print("-" * 50)
    # print("SENZA BUG")
    # print("CORRETTI: ", tot_correct)
    # print("ERRATI: ", tot_error)
    # print("TOTALI: ", tot_tot)
    # tot_correct_percentuale = (tot_correct / tot_tot) * 100
    # tot_error_percentuale = (tot_error / tot_tot) * 100
    # print(tot_correct_percentuale)
    # print(tot_error_percentuale)
    # assert round(tot_correct_percentuale, 2) + round(tot_error_percentuale, 2) == 100.0


if __name__ == "__main__":
    main()
