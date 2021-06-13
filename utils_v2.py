def backline():        
    print('\r', end='') # use '\r' to go back

def progress_bar(epoch_n, epochs):
    if epoch_n > 0:
        backline()
    pctg = ((epoch_n/epochs)*100)
    s = 'Progress: '
    s += '['
    inc = 5
    for p in range(0, 100, inc):
        if p < pctg:
            s += '='
        elif p - inc < pctg:
            s += '>'
        else:
            s += ' '
    s += f'] ({int(pctg)}%)'
    print(s, end='')