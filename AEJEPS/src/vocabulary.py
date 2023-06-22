
ACTION_DESCRIPTION = [
    'forwards', 
    ':POT', 
    ':BLUE-METAL-PLATE', 
    ':SPOON', 
    'move', 
    ':SHOE', 
    'in', ':FORK', 'put', 
    ':BOTTLE', ':GLASSES', ':SPATULA', 
    'to', 'the', ':KNIFE', 'top', ':CEREAL', 
    ':WEISSWURST', ':BREAKFAST-CEREAL', ':GLOVE', 
    ':BUTTERMILK', ':RED-METAL-PLATE', 
    'behind', 'right', ':MUG', ':CAP', ':BOWL', 'of', 'on', 
    ':MONDAMIN', 'shift', 'left', 'front', ':CUBE', ':MILK', 
    ':PLATE', ':CUP', 'backwards'
    ]


MOTOR_COMMANDS = [
    'POSE-4', 
    'POSE-8', 
    ':POT', 
    ':BLUE-METAL-PLATE', 
    'POSE-10', 
    'RED', 
    ':SPOON', 
    ':SHOE', 
    ':FORK', 
    ':BOTTLE', 
    ':GLASSES', 
    ':SPATULA', 
    'POSE-11', 
    'POSE-15', 
    'POSE-5', 
    'POSE-12', 
    'POSE-3', 
    'POSE-14', 
    'NIL', 
    'POSE-9', 
    ':KNIFE', 
    'POSE-2', 
    ':CEREAL', ':WEISSWURST', ':BREAKFAST-CEREAL', ':GLOVE', ':CUBE', 'GREEN', ':BUTTERMILK', ':RED-METAL-PLATE', 
    "#'*leftward-transformation*", "#'*forward-transformation*", "#'*backward-transformation*", 
    'BLUE', ':MUG', ':CAP', ':BOWL', ':MONDAMIN', 'POSE-6', "#'*rightward-transformation*", 
    'POSE-13', 'POSE-7', ':MILK', 'POSE-1', ':PLATE', ':CUP'
    ]

if __name__ == '__main__':
    print("#tokens: ", len(ACTION_DESCRIPTION)+len(MOTOR_COMMANDS))
    print("#tokens: ", len(ACTION_DESCRIPTION))
    print("#tokens: ", len(MOTOR_COMMANDS))