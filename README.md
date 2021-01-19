# automata-groups
My final project for course 3. 

-------------------------------------------------------------------
## Гіпотези / Твердження

#### Зупинки алгоритму

1. Алгоритм перевірки скінченності порядку елемента, що
    - спускається _тільки по вершині 0_ 
    - порівнює зустрічні елементи _як слова_
    - слова _не редукуються_ 
   
     завжди зупиняється. 
   
     __Неправда.__ Контрприклад: `abcfc`
```
Generation: 1, name: abcfc
Generation: 2, name: acfcbfcabc
Generation: 3, name: acbacfbcabcacfcb
Generation: 4, name: abcfcfc
Generation: 5, name: acfcfcbfcabcfc
Generation: 6, name: acfcbfcfcfcfcacfbcabc
Generation: 7, name: acbfcabcabcacfcfcfcfcfcb
Generation: 8, name: abcabcabcacfcfbcfcfcfcfcfcfc
Generation: 9, name: acbacfbcfcfcabcabcabcacfcfcfcfcfcfcfcbacbacbfcfc
Generation: 10, name: abcfcacbacfcfcfbcabc
Generation: 11, name: acfcfcbacbfcacfcb
Generation: 12, name: acfcfbcfcfcabcacfcbacbacbacfbcfcfc
Generation: 13, name: acbfcabcacfcfcfcfbcfcfcabcabcfcfcfcacfbcabcfc
Generation: 14, name: abcabcacfcbfcabcabcfcacbacfcfcbacfbcfcfcfcacbacfcfcfcfcbacfcfcfcfcbfcfcfcfcfcfcacbfcabcabc
Generation: 15, name: acbacbacbfcacfcfcfcfcfcfbcfcfcfcfcfcfcabcabcabcabcabcacfcbacfcfcbacbfcabcfcfcfcfcacfcbfcfcacbacb
Generation: 16, name: abcabcacfcfbcfcfcabcabcabcabcabcacfcbacbacfcfcfcfcfbcabc
Generation: 17, name: acbacbfcabcabcabcabcabcacfcfbcabcabcabcacfcfcfcbacbacbacbacbacfcfcb
Generation: 18, name: abcfcfcfcfcfcfcabcabcacfcbacbacbacbacbacfcfcbacbacbacbacbacfbcabcabcabcacfcbacbacbacbacbacfbcabcabcabcabcabcabcacfcbacbacfbcabcabcfcfc
Generation: 19, name: acfcfcfcfcfcfcbacfbcabcabcfcfcabcabcabcabcabcabcacfcbacbacbacfbcabcabcabcfcfcfcfcabcabcacfcbacbacbacbacbacbacbacbacbacbacbacbacbacbacbfcabcfcfcabcabcacfcbacbacbacbacbacbacbacfbcabcabcfcfcabcfc
Generation: 20, name: acfcfbcfcacbacbacbacbacbacbacbacbacfcfcfcfcbacfbcabcabcabcabcabcabcabcabcfcfcabcabcabcabcabcabcfcacfcfcbacbacbfcabcabcabcabcabcabcacfcbacbfcfcfcabcabcabcabcabcabcabcfcacbacbacbacbacbacbacbacbacbacbfcabc
Generation: 21, name: acbfcabcabcabcabcfcfcfcfcabcabcabcabcabcabcabcabcabcfcacfbcabcabcabcabcabcabcabcacfcfcfcbacbacbacbfcabcabcabcabcabcabcacfcfcfcfbcfcacbacbacbfcfcfcabcfcfcabcfcacbacbacbacbacbacbacbacbacbacbacfbcabcabcabcabcacfcbacbacbacbacbacbfcabcabcabcabcabcabcacfcbacbacbacfbcfcacbacbacbacfcfcb
Generation: 22, name: abcabcabcabcabcfcacbacbacbacbacfcfcbacbacbacbacfbcabcfcfcfcfcfcacbacbacbfcabcfcacbacbacbacbacbacbacbacbacbacbacbacbacbacbacbacbacbacbacbacbacbacbacbacbacbacbfcabcacfcbacbacfcfcfcfcbacbacbacbacbfcfcfcfcfcabcabcabcabcabcabcabcabcacfcbfcabcfcacfcfcbfcabcabcabcabcabcfcfcabcabcabcabcabcabcabcabcabcacfcfcfcbacbacbacbfcfcacbacbacbacbacfcfcbacbacbacfcfcfcfcfcfcbfcabcabcabcabcabcabcabcabcabcabcabcacfcbacbacbacfbcabcfcacbacbacbacfbcfcfcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcacfcbacbacbacbacbacbacbacfbcabcabcfcacfcfcbacbacfbcabcabcfcfcabcabcabcabcfcfc
Generation: 23, name: acbacbacfcfcbacbacbacbacbacfcfcfcfcfcfcbfcabcabcabcabcabcabcabcabcabcabcabcabcabcabcacfcfbcabcabcfcacbacbacbacbacbacfcfcbacbacbacfcfcbacbacbacbacbacfbcabcfcacbacbacbacbacfbcabcfcfcfcfcfcfcfcfcabcabcabcabcabcfcacbacbacbacbacbacbacbacbacfbcabcabcabcfcacfbcabcabcabcfc
Generation: 24, name: abcacfcfcfcbacbacbacbacbacbacbacfbcabcabcabcabcabcfcfcabcacfcbfcabcabcfcfcfcacbacbacfcfcbacbfcfcacbacbacfbcabcabcfcfcfcfcfcfcfcfcfcacbacbacbacbacbacfbcabcfcfcabcabcabcabcabcabcabcabcfcfcfcacbacbacbacbacbacbacbacbacbacbacfcfcbacbfcabcfcfcabcabcacfcfbcabcabcabcabcabcabcabcabcabcabcabcabcabcabcacfcbacfcfcbacbacbacfbcabcabcfcacbacbacbacbacbacfcfcfcfcfcfcfcfcbacbacbfcabcabcabcabcfcacbacbacfcfcbacbacbacbacbacfcfcbacbacbacbacbacbacbacbacbacbfcabcabcacfcbacbacbacbacbacbacfcfcbfcfcfcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc
Generation: 25, name: acfcfcbacbacbfcabcacfcbacfcfcfcfcbacbfcabcfcfcfcfcabcabcabcabcfcfcabcabcabcabcabcfcfcabcabcfcfcfcfcacfcbacfcfcfcfcbacbacbacbacbfcabcfcfcabcabcacfcbacfbcabcabcacfcfcfcbacbacbacbacbacbacbacbacb
Generation: 26, name: acfcbacbacfcfcbfcfcacbacbacfcfcbacfcfcfcfcfbcfcfcfcfcabcabcfcacbacbacbacbacfbcabcabcabcabcfcfcabcfcfcacfcbacbacfcfcfcfcbacbacfcfcbacbacbfcabcabcfcacbacfcfcbfcabcabcacfcbacfcfcbacbacbacfbcfcfcfcfcabcabcfcacbacbfcabcabcabcabcabcfcfcacfcfbcabcabcabcfcfcabcabcabcacfcbacbacbacbacbacbacbacbacbacfbcabcabcacfcfbcfcfcabcabcabcabcfcacbfcfcfcacfcbacbacbacbacbacfcfcbacfbcfcfcfcfcabcabcabcabc
Generation: 27, name: acbacbacfbcabcacfcfbcfcacbfcabcabcfcacfcfcfbcacfcbacbacfbcabcabcabcabcfcfcfcfcacfcfcfcfcfcbacfcfcbacbacbfcacfcbacbfcabcabcabcacfcbacfbcfcacbfcacfcbacbfcfcfcabcfcfcfcacbacbacfbcfcfcfcfcfcfcabcabcabcabcabcabcabcabcabcfcacbacbacfcfcbacbacbacbfcacfcbacfbcfcfcabcabcfcacbfcabcabcabcabcabcabcfcfcfcacbacfbcabcabcabcabcfcfcfcacbacbfcfcacbacbacbacbacbacfbcfcfcabcabcabcabcacfcbfcabcfcfcacfcbfcfcacbacbacbacbacbacbacbfcabcfcfcabcfcfcfcfcabcfcfcabcfcacbacfbcabcabcacfcbacbacbacbfcfcacbacbacbacbacfcfcfbcabcabcabcfcfcabcabcabcabcabcabcabcacfcfcfcbacbacfcfcfcfcfbcabcabcfcfcabcfcacbacb
Generation: 28, name: abcabcabcacfcfcfcbacfcfcbacbacbacbacbacbfcfcfcfcfcfcfcacfcfbcfcfcabcabcacfcfcfcbfcabcabcfcfcabcabcabcabcabcabcabcabcabcabcacfcfbcfcacbfcfcfcabcabcabcabcabcfcacbacbfcabcabcabcfcfcfcacbfcabcabcabcfcacfcfcfcfcbfcabcabcfcfcabcabcabcabcacfcbacbfcabcabcabcabcabcabcabcacfcbacbacfcfcbacfcfcbfcabcabcfcfcfcacbacbfcfcfcfcfcabcfcfcacfcfbcfcfcabcfcfcabcabcabcabcacfcfcfcfbcfcacfcfcfcfcfcfcfcfcfcfcbacbacbacbacbfcabcfcfcabcabcacfcfcfcbacfcfcbacbacbacfcfcfcfcbacbacfcfcfcfcfcfcfcfcbacbacfbcabcfcfcfcacbacbacbacbacbacbacbacfcfcbfcfcacfcfcbacfbcabcfcacbacbacbacbacfbcabcabcabcfcfcfcfcabcfcfcfcfcfcacfcfcbacbfcfcacfbcacfcbacbacfcfcfcfcfcfcbacfbcabcacfcbacbacbacbfcfcfcabcfcfcabcfcfcfcacbacbacfbcabcfcfcabcfcacbacbacbacbfcabcfcfcabcabcabcabcfcacbacbacbacfcfcfbcabcabcabcabcabcfcfcabcfcacbacbacbacbacbacbacbfcabcabcfcfcfcacbacbacbacfcfcfbcabcabcfcacbacbacbacbacbacfbcabcabcabcacfcfbcabcabcabcabcabcfcacfcfcbacbacbacbacbacbacfbcfcacbacbacbfcabcfcacbacbacbfcfcacbacbacbacbacfcfcbacbacbacbacbacbfcabcabcabcabcabcabcabcabcabcfcacbacbacbacbfcabcfcacbacbacbacbacbacbfcabcabcabcabcacfcbfcacfcfcfcbfcabcfcacfcfcbfcabcabcabcacfcfcfcbacbacfcfcbacbacbacbacfbcacfcbacbfcabcabc
Generation: 29, name: acbacfcfcbacbacbacbacbacbfcfcacbfcabcabcacfcbacbfcabcabcabcabcabcabcabcabcabcabcacfcfcfcfcfcbacbacbfcabcabcabcabcfcacbacbacfcfcfbcabcabcfcfcabcabcabcabcabcabcabcabcacfcbacfbcabcabcfcacbacbfcfcacbfcabcfcfcfcfcfcacfcfcfcfcbfcabcabcacfcbacfbcabcfcfcfcfcabcfcfcfcfcfcfcfcfcabcfcfcabcabcabcacfcfcfcbacbacfcfcbacbfcfcfcacfcfcfcfbcacfcfbcfcfcabcabcfcacfcfcfcfcbfcabcabcabcfcacbacbfcabcacfcbacbacbfcabcabcabcabcabcabcabcfcacbacbacbacfbcabcabcabcabcabcfcfcfcacfbcabcabcabcabcabcabcabcabcabcabcacfcbacbacbacbacbfcabcabcabcabcabcabcfcfcfcfcfcfcfcacfbcabcabcabcacfcbacbacfbcabcabcacfcbacbacbacfbcfcfcabcabcabcfcfcacfcfcfcbacfcfcbacfcfcbacbacbacbacbacfbcabcfcacbacbacbacfcfcfcfcbacbfcfcfcfcfcfcacbacbacbacbacbacbacbacfbcacfcbacbfcabcfcfcabcfcfcfcacfcfcbacbacfcfcfcfcfcfcfbcabcabcabcfcfcfcfcacfcfbcacfcfcfcbfcabcabcabcabcfcfcfcacfbcfcacbacbacbacbacbacbacfcfcfcfcbfcfcacfbcfcacbacbacbacfcfcbacbacfbcabcfcacfcfcbfcabcabcabcfcacbacbacbacbacfcfcbacbacfcfcbacbacbacfcfcbfcabcabcabcfcfcabcacfcbacfcfcbacbacbacbacbacfcfcbacbacbacbacbacbacbfcabcfcacbacbacbfcabcabcfcfcabcabcabcabcabcabcabcabcabcabcabcabcabcabcfcacbacbacbacbacbacbacbacbacbacfbcabcacfcbacbacfbcacfcfbcfcfcfcfcacfcfcfcfcfcfcfcfbcfcfcfcfcfcacbacbacbacbacbacbfcfcacbacbacfcfcbacbfcabcabcabcabcacfcbacbfcabcabcabcabcacfcbacbacbacbacbacbacfbcfcfcfcfcabcfcacfcfcfbcfcacbacbacfbcacfcfcfcbacbacbacbacbacfcfcbacfcfcbacbacbacfcfcfcfcbacbacbacfcfcfcfcbfcfcfcabcabcabcabcabcabcabcabcfcacfcfcfcfcfcfcbacbfcfcfcabcfcfcfcfcfcfcabcabcacfcfcfcbfcabcfcacbacbacbacfcfcbfcabcabcabcfcacbacbacbacfbcabcabcabcabcabcfcacbacbacbacbacbacbacbacbfcabcabcfcfcfcacbacbacbacbacbacbacbacfbcabcabcabcabcabcacfcfcfcbfcabcfcacbacbacbacbacfbcabcabcfcacbacbacbacbacfcfcbacbacfbcacfcbacfcfcbacbacfcfcbacbacbacbacbacbacbacb
Generation: 30, name: abcfcfcabcabcabcabcabcabcacfcbacbacbacbacbacfcfcbacbacbacbacbfcabcfcfcfcacbacbacbacbacbacbacbfcabcabcabcfcfcfcfcfcfcfcfcabcabcfcacfcfcfcfcfcfcfcfcbfcabcabcabcacfcbacbacfbcfcacfbcacfcfcfcbacfcfcfbcabcabcabcabcfcfcabcfcacbacbacbacbacbacbacfcfcfcfcbacbacbacbacbacfbcabcabcabcabcabcabcabcabcfcfcabcabcabcabcacfcbacfbcabcfcacbacfcfcfcfcbacfbcabcabcfcacbacbacbacfcfcfcfcfcfcfcfcfbcfcacfcfcfcfcbacbacfcfcbacbacfcfcfcfcfbcacfcbacbacbfcabcabcabcabcacfcbfcfcacbacbacbacfbcabcabcacfcbacbfcabcabcfcfcabcfcfcabcacfcbacbfcabcacfcbacbacbacbacbacfbcabcabcabcabcabcabcabcabcfcacbacbacbacbacbacbacbacbacbacbacfbcabcacfcfcfcfcfcfcfcfbcfcfcabcabcabcabcacfcbacbacfbcfcfcabcabcabcfcfcabcacfcfcfcfcfcbacfbcabcfcfcfcfcabcacfcbfcabcabcabcabcabcabcabcabcacfcfbcfcacfcfcfcfcfcfcbacfcfcbfcabcacfcbacbfcabcabcabcabcabcabcabcabcabcabcabcfcfcabcabcabcabcabcabcabcabcabcacfcbacfcfcbacfcfcbacbacbacbacfbcacfcbacbacbacbacbacbfcfcfcabcabcabcabcabcabcabcabcabcabcabcacfcfbcabcabcabcabcabcabcacfcbacfcfcbacbacbacbacfbcfcacbacbfcfcfcacfcbacbacbacbacfcfcfcfcbfcfcfcabcfcfcfcfcabcfcfcabcfcfcfcfcfcfcfcacbfcfcfcfcfcfcacbacbacfbcabcabcabcabcabcabcabcabcabcabcfcfcfcfcabcabcfcfcfcacbacbacfbcabcabcabcacfcfcfcbacbfcacfcbacfbcfcfcabcabcabcabcabcabcfcfcfcfcabcfcfcabcabcabcabcacfcbfcabcfcacfbcacfcfbcabcabcabcfcacbacfbcabcabcabcabcfcacbfcabcabcabcfcfcfcfcfcacbfcabcacfcbfcfcfcfcacbacbacbacbacfbcacfcbacbacfcfcbacbacfbcabcabcfcfcabcabcabcfcacbacbacbacbfcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcfcfcabcfcfcfcfcacfcfcfcfcfcfcfcfcfcfcfcbacbacbacbacbacbacbacbacbacbacbacbacbacbacbfcfcacfbcabcabcacfcbacbacbacbacbacfbcfcfcabcacfcbacbacbacfcfcfcfcbacbacbacbacfcfcfcfcfcfcbfcfcacbacfbcabcabcabcfcfcfcacbacbacbacbacbacbacfcfcbacfcfcfcfcbacbacbacfbcabcabcabcabcabcabcabcabcacfcfbcfcfcabcfcfcabcabcabcabcacfcfcfcbacfbcfcfcfcfcfcfcabcfcacbacbacfbcabcabcfcfcabcabcabcabcfcacbacfcfcfcfcfcfcfbcabcabcacfcbfcfcfcfcacfcfcbacbacfbcacfcfcfcfcfcfbcfcacfcfcbacbfcabcabcacfcbacbacbacbfcabcabcabcabcabcabcabcfcacbacbacbacbacbacbacbacbacbacbacbacbacbacbfcfcfcfcfcabcfcfcabcabcfcacfbcfcfcacfcbfcabcacfcbacbfcfcacbacbacbacbacbacbacbacbacbacbacfcfcbfcacfcfcfcbacbfcfcfcfcfcfcfcfcfcfcacbacbacbacbacbacbacfcfcfcfcfcfcbacbacbacfcfcbacbacfcfcbacbacfbcabcfcfcfcacfbcfcfcabcabcacfcbfcabcfcacbacbacbacbacbacbacbfcabcabcabcabcabcabcacfcfbcfcacfcfcbfcfcacbacbacbacbacbacbfcabcfcfcabcabcabcabcabcacfcbacbacfbcabcabcabcfcacfcfcbfcabcfcfcfcfcabcabcacfcbacbacbacfcfcfbcfcacfcfcbacbfcabcfcfcabcabcacfcbacfcfcbacbacfcfcbacbacbfcabcabcabcabcabcabcfcacbacbacbacbacbacbacbacbacbacbacfcfcbfcabcabcfcacbacbacbacbacfbcabcacfcbacbacfbcabcabcabc
Generation: 31, name: acfcfcbacbacbacfbcabcabcfcfcabcabcfcfcabcabcabcfcfcfcacbfcfcfcfcfcfcfcfcfcfcfcabcfcacbacfbcfcfcfcacfcfcfbcabcabcfcacbacbacbacbacfcfcfbcfcacbacbacbacfcfcfbcacfcbacbacfbcacfcbacbacbacbacbacbacbfcabcabcabcabcacfcbacfcfcbfcabcacfcbacbacbacbacbacbacbacbacbacbacbfcabcabcabcabcabcfcfcfcfcfcfcfcfcfcacbacbacbacbacbfcabcabcabcfcfcfcfcfcfcabcabcfcacfbcabcabcabcabcabcabcabcabcacfcfcfcfcfcbacfbcfcacbacbacbacbacfbcfcfcacfcfbcabcabcfcacbacbacbacbacbacfbcabcabcabcabcabcabcacfcbacbacbacbacbfcabcfcfcabcabcfcfcfcfcfcacfcfcfcfcbfcabcfcfcabcfcfcabcfcfcabcabcfcacbacbacbacbacfcfcfbcfcacbacbacbfcfcacbacbacbacfcfcfbcabcabcabcabcfcacbacbacfcfcfcfcfcfcbacbfcabcabcabcacfcbacbacbacbfcabcabcabcabcabcfcacfcfcfcfcfcfcfcfcbacbacbacbacbacbacbacbacbacbacbacbacbacbacbfcfcfcabcabcabcfcfcabcacfcbacbacbacbacfcfcbfcabcabcabcabcfcacbacbacbacbacbacbacfbcfcfcfcfcabcabcabcabcabcabcabcabcabcabcacfcfcfcbfcabcabcabcabcacfcbacbfcfcacfcfcbacfcfcbacbacfcfcfcfcbacfbcfcacfbcabcacfcfbcacfcbacfbcabcfcacbacbacbacbacbacbacbacbacbacbacbacbacbacbfcfcacbfcfcacbacfbcfcacbacbacbacbacbacbacbacbacbacbacfbcacfcbacbfcfcfcabcabcabcacfcfbcabcfcfcabcfcfcabcfcfcfcacbacbacfcfcbacbacbacfbcacfcfcfcbfcabcabcabcabcabcacfcbacbfcfcfcfcfcabcabcacfcbfcfcfcabcabcfcfcacfcbacbacbacbacbacbfcabcabcabcabcabcfcfcfcacbacbacbacbacbacfbcabcabcabcabcabcfcfcabcabcacfcbfcabcabcabcabcabcabcabcfcfcfcacfcfcfbcabcabcabcacfcfcfcfcfcbacbacbfcabcabcabcabcacfcbacbacbacbacbacbacbacbacbacbfcabcabcabcabcacfcbfcacfcbacbacbfcabcacfcfcfcfcfcfbcacfcfbcacfcbacbacfbcabcabcfcfcabcabcabcfcacfbcfcfcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcacfcfcfcfcfcbacbacfbcabcfcacfcfcbacbacfcfcfcfcbacbacbacbacbacbacbfcfcfcfcfcfcfcacfcbacbacbacbacbacbacbacbacfcfcbacbacbacbacbacbacfbcfcfcabcabcfcfcabcabcabcfcfcfcfcacfcfcfcfcfcfbcabcacfcbfcabcfcacfcfcbfcfcfcfcfcabcabcabcabcabcabcabcabcabcabcabcabcfcacbfcabcabcabcabcabcacfcbacbacbacbfcabcabcabcabcabcabcfcacfcfcbacbacfbcabcabcacfcbacbfcabcabcabcabcabcabcabcabcabcfcfcfcfcfcfcabcabcfcfcabcfcfcabcfcacbacfcfcbacbacbacbacbacbacbacbacbacbfcabcfcacfcfcfcfcfcfcbacfbcabcabcfcacfbcabcfcfcfcfcabcabcfcfcfcfcfcfcfcacbacbacbacfcfcfcfcbacfcfcbacbacbacbacfbcfcacbacbacfbcfcfcfcacbacbacbacbfcabcabcabcabcabcfcfcfcfcfcfcfcfcfcfcfcfcabcfcfcfcfcfcfcfcacfbcfcfcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcfcfcabcfcacbfcacfcbacbacbfcabcabcabcabcabcfcfcfcfcfcfcabcfcfcfcacbacbacbacbacbacbacfcfcbacbacbacfbcacfcbfcabcfcfcfcacbacbacbacbacbacbacbacbacbacbacbfcfcfcfcacbacbacbacbacbacbacfcfcbacbacbacbacbacbacbacfcfcbacfcfcfcfcbacfbcabcfcfcfcacfbcfcacbacbacfbcacfcbacbacbacfcfcbacbfcabcabcfcfcabcfcacbacbacbacbacbacbacbacbacfbcabcabcabcfcacbacbacbacbacbacbacbacbacfcfcfcfcfcfcfcfcbacfcfcfcfcbacbacfbcabcabcacfcbacfbcabcabcabcabcfcacbacbacbacbacbacbacfcfcbacbacbacbacfcfcbacbacfbcfcacbacfcfcbacfcfcfcfcbfcfcfcfcfcabcfcfcabcfcfcfcfcfcfcabcfcfcfcacbacbacbacbacbacbacbacbfcabcfcfcabcabcacfcbacbacbacbacfcfcbacfcfcbfcabcabcabcabcacfcfcfcbacbfcabcacfcfbcfcfcabcacfcbacbacbacbacfbcacfcfbcfcfcfcfcabcabcabcabcabcabcabcabcabcabcabcabcfcfcacfcbacbacbacbacbacbacbacbacbacbacbfcabcabcabcabcabcabcabcabcabcabcabcacfcbacbacbacfbcfcfcabcabcfcacbacbfcacfcfbcfcfcabcfcfcfcacbfcfcacbacbacbacbacbacbacbacbfcfcfcabcfcfcfcfcabcacfcfcfcbacbacbacfcfcfcfcbfcabcabcabcabcacfcbfcfcfcfcacbacbacbacbfcfcfcabcabcacfcfcfcfcfcfbcacfcbacfcfcbacbfcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcfcfcacfcfcfcfbcabcabcabcabcabcabcabcabcabcabcacfcfcfcbacbacbacbacfcfcfcfcfcfcbacbfcabcabcabcacfcfbcabcfcfcfcacfcfcbacbacfcfcfcfcfcfcfcfcbfcabcfcacbacbfcacfcfbcabcabcacfcfcfcfcfcbacbacbacfcfcfcfcbacbacbacbacbacbacbacbacbacbacfcfcfcfcfcfcbfcabcabcabcfcfcfcfcabcabcabcabcabcabcacfcfcfcfcfcfcfcfcfcfcfcfcfcfcfcbacbacfbcabcabcfcacbfcabcabcacfcbfcabcabcabcfcfcfcacfbcabcabcabcfcacbacbacfbcabcabcabcabcacfcbfcfcfcabcacfcfcfcbacbacfcfcbacfbcfcfcabcfcfcabcfcacbacbacbacbacbacbacbacbacbacbacfbcabcabcabcabcabcabcacfcbacb
<maximum deep exception>
```

2. Алгоритм перевірки скінченності порядку елемента, що 
    - спускається _тільки по вершині 0_
    - порівнює зустрічні елементи _як слова, з точністю до циклічних зсувів_
    - слова _не редукуються_ 
    
     завжди зупиняється.
   
     __Неправда.__ Контрприклад: `abcfc`
   

3. Алгоритм перевірки скінченності порядку елемента, що 
    - спускається _тільки по вершині 0_ 
    - порівнює зустрічні елементи _як слова, з точністю до циклічних зсувів_
    - слова _редукуються_
    
     завжди зупиняється.
   
     __Неправда.__ Контрприклад: `abcfc`


4. Алгоритм перевірки скінченності порядку елемента, що 
    - спускається _тільки по вершині 0_
    - порівнює зустрічні елементи _розв'язуючи проблему слів_
    - слова _редукуються_
    
    завжди зупиняється.
    
    __Неправда.__ Той же контрприклад -  `abcfc`


5. Алгоритм перевірки скінченності порядку елемента, що 
   - спускається _тільки по основних гілках_ (тобто 
     завжди по 0 + завжди по 1 + завжди по 2 + завжди по 3)
     _пошуком в ширину_
   - порівнює зустрічні елементи _як слова_
   - слова не редукуються
   
   завжди зупиняється.
   
   __Невідомо.__


6. Алгоритм перевірки скінченності порядку елемента, що 
    - спускається по всіх елементах _пошуком в ширину_
    - порівнює зустрічні елементи _як слова_
    - слова _не редукуються_
    
    завжди зупиняється.
   
    __В процесі.__ Поки наче зупиняється

#### Числові характеристики алгоритму

1. Для кожного слова довжини `n` алгоритм знаходить цикл 
    на глибині `O(n)`.
   
    __В процесі.__


#### Інше

1. Існують елементи нескінченного порядку, для яких множина 
   орбітальних станів скінченна, так ще й складається з одного
   елемента.
   
   __Приклад:__ `cfcf` 

![cfcf](./graphs/cfcf.png)

2. Всі порядки - парні числа.

   __В процесі.__ Перевірено елементи довжини <= 7.

3. 