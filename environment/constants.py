ALL_POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
POWERS_TO_ACRONYMS = {
    'AUSTRIA': 'AUS',
    'ENGLAND': 'ENG',
    'FRANCE': 'FRA',
    'GERMANY': 'GER',
    'ITALY': 'ITA',
    'RUSSIA': 'RUS',
    'TURKEY': 'TUR'
}

ACRONYMS_TO_POWERS = {
    'AUS': 'AUSTRIA',
    'ENG': 'ENGLAND',
    'FRA': 'FRANCE',
    'GER': 'GERMANY',
    'ITA': 'ITALY',
    'RUS': 'RUSSIA',
    'TUR': 'TURKEY'
}

POWER_ACRONYMS_LIST = ['AUS', 'ENG', 'FRA', 'GER', 'ITA', 'RUS', 'TUR']

LOCATIONS = ['YOR', 'EDI', 'LON', 'LVP', 'NTH', 'WAL', 'CLY',
             'NWG', 'ENG', 'IRI', 'NAO', 'BEL', 'DEN', 'HEL',
             'HOL', 'NWY', 'SKA', 'BAR', 'BRE', 'MAO', 'PIC',
             'BUR', 'RUH', 'BAL', 'KIE', 'SWE', 'FIN', 'STP',
             'STP/NC', 'GAS', 'PAR', 'NAF', 'POR', 'SPA', 'SPA/NC',
             'SPA/SC', 'WES', 'MAR', 'MUN', 'BER', 'BOT', 'LVN',
             'PRU', 'STP/SC', 'MOS', 'TUN', 'LYO', 'TYS', 'PIE',
             'BOH', 'SIL', 'TYR', 'WAR', 'SEV', 'UKR', 'ION',
             'TUS', 'NAP', 'ROM', 'VEN', 'GAL', 'VIE', 'TRI',
             'ARM', 'BLA', 'RUM', 'ADR', 'AEG', 'ALB', 'APU',
             'EAS', 'GRE', 'BUD', 'SER', 'ANK', 'SMY', 'SYR',
             'BUL', 'BUL/EC', 'CON', 'BUL/SC']

DAIDE_LOCATIONS = ['YOR', 'EDI', 'LON', 'LVP', 'NTH', 'WAL', 'CLY', 'NWG', 'ECH', 'IRI', 'NAO', 'BEL', 'DEN', 'HEL',
                   'HOL', 'NWY', 'SKA', 'BAR', 'BRE', 'MAO', 'PIC', 'BUR', 'RUH', 'BAL', 'KIE', 'SWE', 'FIN', 'STP',
                   '(STP NCS)', 'GAS', 'PAR', 'NAF', 'POR', 'SPA', '(SPA NCS)', '(SPA SCS)', 'WES', 'MAR', 'MUN', 'BER',
                   'GOB', 'LVN', 'PRU', '(STP SCS)', 'MOS', 'TUN', 'GOL', 'TYS', 'PIE', 'BOH', 'SIL', 'TYR', 'WAR',
                   'SEV', 'UKR', 'ION', 'TUS', 'NAP', 'ROM', 'VEN', 'GAL', 'VIE', 'TRI', 'ARM', 'BLA', 'RUM', 'ADR',
                   'AEG', 'ALB', 'APU', 'EAS', 'GRE', 'BUD', 'SER', 'ANK', 'SMY', 'SYR', 'BUL', '(BUL ECS)', 'CON',
                   '(BUL SCS)']

UNIT_TYPES = ['A', 'F']
UNIT_TYPES_TO_DAIDE = {
    'A': 'AMY',
    'F': 'FLT'
}
DAIDE_TO_UNIT_TYPES = {
    'AMY': 'A',
    'FLT': 'F'
}

LAND_TYPES = ["LAND", "WATER", "COAST"]

# Order types
WAIVE = 0
HOLD = 1
MOVE = 2
SUPPORT_HOLD = 3
SUPPORT_MOVE = 4
CONVOY = 5
CONVOY_TO = 6
RETREAT_TO = 7
DISBAND = 8
BUILD_ARMY = 9
BUILD_FLEET = 10
REMOVE = 11

# Seasons
SPRING_MOVES = 0
SPRING_RETREATS = 1
FALL_MOVES = 2
FALL_RETREATS = 3
WINTER_ADJUSTMENTS = 4
COMPLETED = 5


