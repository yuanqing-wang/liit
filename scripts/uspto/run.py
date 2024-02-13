from ast import parse
from sling.data.uspto import parse_patent


def main():
    root = parse_patent("_data/grants/2016/I20160105.xml")
    print(root)

if __name__ == "__main__":
    main()