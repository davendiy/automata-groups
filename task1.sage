
G = SymmetricGroup(3)
CHUNK = 400

RECURSION_MAX_DEEP = 20


class NotCalculatedError(Exception):
    pass


class BadExpressionError(Exception):
    pass


class NewGroupElement:

    DEFINED_ELEMENTS = {"e": ''}

    @staticmethod
    def from_cache(name):
        if name in NewGroupElement.DEFINED_ELEMENTS:
            return NewGroupElement.DEFINED_ELEMENTS[name]
        else:
            raise NotCalculatedError()

    def __init__(self, name="e", perm=G(""), el_list=("e", "e", "e"), primitive=True):
        self.prim = primitive

        if self.prim:
            self.name = "e"
            self.perm = G("")
            self.el_list = ("e", "e", "e")
            NewGroupElement.DEFINED_ELEMENTS['e'] = self

        else:
            if name in NewGroupElement.DEFINED_ELEMENTS:
                tmp = NewGroupElement.DEFINED_ELEMENTS[name]
                self.name = tmp.name
                self.perm = tmp.perm
                self.el_list = tmp.el_list
            else:
                assert isinstance(perm, sage.groups.perm_gps.permgroup_element.SymmetricGroupElement), "bad type of permutation"
                assert isinstance(el_list, list) or isinstance(el_list, tuple), "bad type of el_list"
                assert len(el_list) == 3, "bad lenght of el_list"
                assert isinstance(name, str), "bad name"
                for el in el_list:
                    if el not in NewGroupElement.DEFINED_ELEMENTS and el != name:
                        NewGroupElement.parse_str(el)

                self.name = name
                self.perm = perm
                self.el_list = el_list

                NewGroupElement.DEFINED_ELEMENTS[self.name] = self

                if self.is_primitive():
                    self.perm = G("")
                    self.el_list = ("e", "e", "e")
                    self.prim = True
                    NewGroupElement.DEFINED_ELEMENTS[self.name] = self

    def __str__(self):
        return self.name + " = " + str(self.perm) + " (" + ', '.join(self.el_list) + ")"

    def __repr__(self):
        return self.name

    def __call__(self, word):
        if self.prim:
            return word

        elif not word:
            return []
        elif len(word) == 1:
            return [self.perm(word[0])]
        else:
            el = self.el_list[int(word[0]) - 1]
            return [self.perm(word[0])] + NewGroupElement.DEFINED_ELEMENTS[el](word[1:])

    def __mul__(self, other):
        if self.prim:
            return other

        if other.prim:
            return self

        res_name = self.name + other.name
        res_perm = other.perm * self.perm

        res_els = []
        for i in xrange(1, 4):
            tmp1 = self.el_list[other.perm(i) - 1]
            tmp2 = other.el_list[i - 1]
            tmp_res = tmp1 + tmp2
            tmp_res = tmp_res.replace('aa', '').replace('bb', '').replace('cc', '').replace('e', '')
            tmp_res = tmp_res if tmp_res else "e"
            res_els.append(tmp_res)

        for el in res_els:
            NewGroupElement.parse_str(el)
        res = NewGroupElement(res_name, res_perm, res_els, primitive=False)
        return res

    def __pow__(self, pow):
        res = NewGroupElement()
        for i in range(pow):
            res *= self
        return res

    def is_primitive(self, checked=()):
        if not checked:
            checked = set()
        if self.name in checked:
            return True
        if self.prim:
            return self.prim

        if self.perm != G(""):
            return False

        succ = True
        checked.add(self.name)
        for el in self.el_list:
            tmp = NewGroupElement.DEFINED_ELEMENTS[el]
            succ = succ and tmp.is_primitive(checked)
            if not succ:
                break
        return succ

    def order(self):
        if self.prim:
            return 1

        perm_order = self.perm.order()
        tmp = self ** (perm_order)
        if (tmp.el_list[0] == tmp.el_list[0][::-1]
                and tmp.el_list[1] == tmp.el_list[1][::-1]
                and tmp.el_list[2] == tmp.el_list[2][::-1]):
            return perm_order * 2 if not tmp.prim else perm_order
        else:
            return float('inf')

    def order2(self):
        checked = set()
        tmp_res = self._order2_helper(checked, 0)
        return tmp_res if tmp_res else float('inf')

    def _order2_helper(self, checked, deep):
        if self.name in checked and not self.prim:
            return 0
        elif self.prim:
            return 1
        elif deep > RECURSION_MAX_DEEP:
            return 0

        res = self.perm.order()
        tmp = self ** res

        checked.add(self.name)

        # print "\ndeep:", deep, "element:", self.get_full_str(), "going to check:", tmp.get_full_str(), "checked:", checked
        # raw_input("press enter to continue...")

        lcm_ord = 1
        for el in tmp.el_list:
            tmp = NewGroupElement.parse_str(el)
            tmp_ord = tmp._order2_helper(checked, deep+1)
            lcm_ord = lcm(lcm_ord, tmp_ord)
            if not lcm_ord:
                break
        return res * lcm_ord

    def order_brute(self):
        tmp = NewGroupElement()
        res = float('inf')
        for i in range(CHUNK):
            tmp *= self
            if tmp.prim:
                res = i
                break
        return res

    @staticmethod
    def parse_str(expression):
        if expression in NewGroupElement.DEFINED_ELEMENTS:
            return NewGroupElement.DEFINED_ELEMENTS[expression]

        for el in set(expression):
            if el not in NewGroupElement.DEFINED_ELEMENTS:
                raise BadExpressionError()

        res = NewGroupElement()
        for el in expression:
            res = res * NewGroupElement.from_cache(el)
        return res


def mul(tmp1, tmp2):
    res = tmp1 * tmp2
    print res.get_full_str()
    return res

e = NewGroupElement()
a = NewGroupElement(name='a', perm=G("(1, 2)"), el_list=['e', 'e', 'a'], primitive=False)
b = NewGroupElement(name='b', perm=G("(1, 3)"), el_list=['e', 'b', 'e'], primitive=False)
c = NewGroupElement(name='c', perm=G("(2, 3)"), el_list=['c', 'e', 'e'], primitive=False)


# interesting = ["ababcbabac", "abacababcb", "abacacbcac", "abcbabacab",
#                "abcbcacbcb", "acababcbab", "acabacacbc", "acacbcacab",
#                "acbcacabac", "acbcbabcbc", "babacababc", "babcbabaca",
#                "babcbcacbc", "bacababcba", "bacacbcaca", "bcacabacac",
#                "bcacbcbabc", "bcbabacaba", "bcbabcbcac", "bcbcacbcba",
#                "cababcbaba", "cabacacbca", "cacabacacb", "cacbcacaba",
#                "cacbcbabcb", "cbabacabab", "cbabcbcacb", "cbcacabaca",
#                "cbcacbcbab", "cbcbabcbca"]
#
# new_interesting = {3: []}
#
# for el in interesting:
#     tmp = NewGroupElement.parse_str(el)
#     now_gen.append(tmp)
#     new_interesting[3].append(tmp)
#
#
# for el in now_gen:
#     for el2 in now_gen:
#         tmp = el * el2
#         tmp_order = tmp.order2()
#         if tmp_order not in {1, 2, float('inf')}:
#             print "order of [", tmp, "] =", tmp_order
#             if tmp_order not in new_interesting:
#                 new_interesting[tmp_order] = []
#
#             new_interesting[tmp_order].append(tmp)




# for i in range(20):
#     print "i:", i
#     next_gen = []
#     for el in gen:
#         for el2 in now_gen:
#             tmp = el * el2
#             tmp_order = tmp.order2()
#
#             if tmp_order not in {1, 2, float('inf')} and tmp.name not in checked:
#                 print "order of [", tmp.get_full_str(), "] =", tmp_order
#                 interesting.append(tmp)
#
#             if tmp.name not in checked:
#                 next_gen.append(tmp)
#                 checked.add(tmp.name)
#
#     now_gen = next_gen
