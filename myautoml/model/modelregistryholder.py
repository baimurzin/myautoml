class ModelRegistryHolder(type):
    REGISTRY = {
        "classification": {},
        "regression": {}
    }

    def __new__(cls, clsname, superclasses, attributedict):
        newclass = type.__new__(cls, clsname, superclasses, attributedict)
        # condition to prevent base class registration
        if superclasses:
            if "classification" in newclass.__module__:
                cls.REGISTRY['classification'][newclass.__name__] = newclass
            else:
                cls.REGISTRY['regression'][newclass.__name__] = newclass
        return newclass

    @classmethod
    def get_registry(mcs):
        return dict(mcs.REGISTRY)
