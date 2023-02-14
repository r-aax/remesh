class Seq:
    """
    Sequential object.
    """

    def get_pred(self):
        """
        Get pred id.

        Returns
        -------
        int
            Pred id.
        """

        pass

    def get_succ(self):
        """
        Get succ id.

        Returns
        -------
        int
            Succ id.
        """

        pass

    def is_loop(self):
        """
        Check if is loop.

        Returns
        -------
        True - if it is a loop,
        False - if it is not a loop.
        """

        return self.get_pred() == self.get_succ()

    def flip(self):
        """
        Flip.
        """

        pass

    def get_els(self):
        """
        Get all elements.

        Returns
        -------
        [Element]
            Elements.
        """

        pass


class Element(Seq):
    """
    Single element.
    """

    def __init__(self, pred, succ, obj):
        """
        Constructor.

        Parameters
        ----------
        pred : int
            Pred.
        succ : int
            Succ.
        obj : object
            Any object.
        """

        self.pred = pred
        self.succ = succ
        self.obj = obj

    def __repr__(self):
        """
        Representation.

        Returns
        -------
        str
            Representation.
        """

        return f'[{self.pred} - {self.succ}]'

    def get_pred(self):
        """
        Get pred.

        Returns
        -------
        int
            Pred.
        """

        return self.pred

    def get_succ(self):
        """
        Get succ.

        Returns
        -------
        int
            Succ.
        """

        return self.succ

    def flip(self):
        """
        Flip.
        """

        self.pred, self.succ = self.succ, self.pred

        if not self.obj is None:
            self.obj.node1, self.obj.node2 = self.obj.node2, self.obj.node1

    def get_els(self):
        """
        Get all elements.

        Returns
        -------
        [Element]
            Elements list.
        """

        return [self]


class Path(Seq):
    """
    Path.
    """

    def __init__(self, e):
        """
        Constructor from single element.

        Parameters
        ----------
        e : Element
            Element.
        """

        self.els = [e]

    def __repr__(self):
        """
        Representation.

        Returns
        -------
        str
            Representation.
        """

        if self.is_loop():
            pref = 'L: '
        else:
            pref = 'P: '

        return pref + ' '.join(map(lambda e: e.__repr__(), self.els))

    def get_pred(self):
        """
        Get pred.

        Returns
        -------
        int
            Pred.
        """

        return self.els[0].get_pred()

    def get_succ(self):
        """
        Get succ.

        Returns
        -------
        int
            Succ.
        """

        return self.els[-1].get_succ()

    def flip(self):
        """
        Flip.
        """

        self.els.reverse()

        for e in self.els:
            e.flip()

    def get_els(self):
        """
        Get all elements.

        Returns
        -------
        [Element]
            Elements list.
        """

        return self.els

    def can_add(self, s):
        """
        Check if it is possible to add seq member to path.

        Parameters
        ----------
        s : Seq
            Sequential object.

        Returns
        -------
        True - if it is possible to add,
        False - if it is not possible to add.
        """

        if self.is_loop():
            return False

        cur = [self.get_pred(), self.get_succ()]
        can = (s.get_pred() in cur) or (s.get_succ() in cur)

        return can

    def add(self, s):
        """
        Add sequantial member.

        Parameters
        ----------
        s : Seq
            Sequential member.
        """

        assert self.can_add(s)

        if self.get_succ() == s.get_pred():
            self.els = self.els + s.get_els()
        elif self.get_pred() == s.get_succ():
            self.els = s.get_els() + self.els
        else:
            s.flip()
            if self.get_succ() == s.get_pred():
                self.els = self.els + s.get_els()
            elif self.get_pred() == s.get_succ():
                self.els = s.get_els() + self.els
            else:
                raise Exception('internal error')

    def rot(self, i):
        """
        Rotate loop path to position i.
        After rotation element with i-th index become the first.

        Parameters
        ----------
        i : int
            Index for rotate.
        """

        assert self.is_loop()

        self.els = self.els[i:] + self.els[:i]


class Border:
    """
    Border (can contain several seq members).
    """

    def __init__(self):
        """
        Constructor.
        """

        self.paths = []

    def print(self):
        """
        Print on screen.
        """

        print(f'Border (count = {len(self.paths)}):')

        for p in self.paths:
            print(p)

    def add(self, e):
        """
        Add new element.

        Parameters
        ----------
        e : Element
            Element.
        """

        li = [p for p in self.paths if p.can_add(e)]
        cli = len(li)

        if cli == 0:
            self.paths.append(Path(e))
        elif cli == 1:
            li[0].add(e)
        elif cli == 2:
            li[0].add(e)
            self.paths.remove(li[1])
            li[0].add(li[1])
        else:
            raise Exception('internal error')
