# EquiViT - Build and test equivariant Vision Transformers


`equivit` is a library based on PyTorch for working with equivariant vision transformers.
It provides implementations of group-equivariant versions of ViT layers that
are not specific to the choice of symmetry group, generalizing previous works on
[$\mathbb{Z}_2$-equivariant ViTs](https://github.com/georg-bn/flopping-for-flops) and 
[$D_4$-equivariant ViTs](https://github.com/davnords/octic-vits) to arbitrary groups. 



### Quick Install
```
python -m pip install -e .
```


### Components
- `equivit.geometry`
    - `groups`: lightweight algebra backend that deals with group theory and representation theory
    - `lattices`: implements common group actions (e.g. $D_4$ on square lattice and $D_6$ on hexagonal lattice)

- `equivit.nn`: group-equivariant ViT layers



    

