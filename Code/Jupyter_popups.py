"""
Author: Martin Schiemer
sends popups as reminder for taking notes
"""

# taken and adapted from https://www.stefaanlippens.net/jupyter-notebook-dialog.html
from IPython.display import display, Javascript

def pop_up_note_reminder():
    
    display(Javascript("""
    require(
        ["base/js/dialog"], 
        function(dialog) {
            dialog.modal({
                title: 'Please take note',
                body: 'Please note down the maximum values for mutual information for the estimators and fill in the respective cells before you start early/perfect stop' ,
                buttons: {
                    'Close': {}
                }
            });
        }
    );
    """))