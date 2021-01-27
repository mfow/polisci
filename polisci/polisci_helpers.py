import os


__polisci_dir = os.path.dirname(os.path.abspath(__file__))


def get_polisci_path(*paths):
    return os.path.join(__polisci_dir, *paths)
