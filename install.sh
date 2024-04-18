
echo $'\n'"Installing packages ..."
pip install -r requirements.txt

echo $'\n'"Installing FIM ..."
pip install -e .

echo $'\n'"Installing pre-commit..."
pre-commit install
cara/.gitattributes
echo $'\n'"Installing git lfs..."
git lfs install
