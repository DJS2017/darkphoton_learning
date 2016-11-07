#! /usr/bin/env python
# This file contains functions for cut.


def OptimalThreshold(pdf_background, pdf_signal, learning_rate=0.01, start=0.0):
    """
    Find optimal threshold given background and signal pdf.
    
    Input:
    ----------
    pdf_background: pdf of background, it should contain the score method,
                    pdf_background.score(x) returns the probability density
                    in x point.
    
    pdf_signal:     pdf of signal, it should contain the score method,
                    pdf_signal.score(x) returns the probability density
                    in x point.
                    
    learning_rate:  the step length for gradient descent

    start:          start point for searching
    
    
    Output:
    ----------
    threshold:     The optimal threshold.
    
    """
    
    # Loss function for minimization
    def Loss(threshold_opt):
        return abs(pdf_background.score(threshold_opt) - pdf_signal.score(threshold_opt))

    
    # initialization
    loss_list = [100]
    threshold_opt = start
    loss = Loss(threshold_opt)
    loss_list.append(loss)
    
    
    # find minimum using gradient descent
    while(loss_list[-2] > loss_list[-1]):
        # look left and right
        threshold_left = threshold_opt - learning_rate
        threshold_right = threshold_opt + learning_rate
        loss_left = Loss(threshold_left)
        loss_right = Loss(threshold_right)
        
        # cannot find lower value
        minimum = min(loss_left, loss_right, loss_list[-1])
        if(minimum == loss_list[-1]):
            return threshold_opt
        
        elif(minimum == loss_left):
            threshold_opt = threshold_left
            loss_list.append(loss_left)
            
        else:
            threshold_opt = threshold_right
            loss_list.append(loss_right)


