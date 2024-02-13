import re
import glob
import torch
from torch.utils.data import Dataset
import pandas as pd
import xml.etree.ElementTree as ET

def parse_patent(path):
    """Parse a USPTO patent XML file and return a dictionary of its contents.

    Parameters
    ----------
    path : str
        The path to the USPTO patent XML file.
    """
    # initialize the results list
    results = []

    # Parse a USPTO patent XML file and return a dictionary of its contents.
    with open(path, "r") as f:
        xml = f.read()

    # delete the namespace
    xml = re.sub(r" xmlns=\".*?\"", "", xml)
    xml = re.sub(r"dl:", "", xml)

    # parse the XML
    root = ET.fromstring(xml)

    # extract the reaction SMILES and product SMILES
    for reaction in root.findall("reaction"):
        reaction_smiles, smiles, paragraph = None, None, None

        # extract the reaction SMILES
        reaction_smiles = reaction.find("reactionSmiles").text

        # extract the product SMILES
        product = reaction.find("productList").find("product")
        if product is not None:
            identifier = product.find("identifier")
            if identifier is not None:
                if identifier.attrib["dictRef"] == "cml:smiles":
                    smiles = identifier.attrib["value"]

        # extract paragraph
        paragraph = reaction.find("source").find("paragraphText").text

        if reaction_smiles is not None and smiles is not None and paragraph is not None:
            results.append(
                [reaction_smiles, smiles, paragraph]
            )

    return results

if __name__ == "__main__":
    import glob
    paths = glob.glob("_data/grants/*/*.xml")
    data = [parse_patent(path) for path in paths]
    data = [item for sublist in data for item in sublist]
    df = pd.DataFrame(data, columns=["reaction", "product", "paragraph"])
    df.to_csv("uspto.csv", index=False)
