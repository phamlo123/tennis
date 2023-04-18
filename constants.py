class Colors:
    GREEN = (82, 126, 45)
    BLACK = (0, 0, 0)
    LIGHT_GRAY = (214, 214, 214)
    PINK = (240, 128, 128)
    BLUE = (117, 128, 240)
    WHITE = (236, 236, 236)
    DARK_GRAY = (74, 74, 74)
    table = {
        GREEN: 0,
        BLACK: 1,
        LIGHT_GRAY: 2,
        PINK: 3,
        BLUE: 4,
        WHITE: 5,
        DARK_GRAY: 6,
    }
    
    @staticmethod
    def lookup(color):
        return Colors.table.get(color, -1)
    
    @staticmethod
    def name(colorId):
        if colorId < 0 or colorId >= len(Colors.table):
            return 'UNKNOWN'
        names = [
            'GREEN', 'BLACK', 'LIGHT_GRAY', 'PINK', 'BLUE', 'WHITE', 'DARK_GRAY'
        ]
        return names[colorId]