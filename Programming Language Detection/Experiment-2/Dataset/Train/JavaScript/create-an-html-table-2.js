(() => {
    'use strict';

    // HTML ---------------------------------------------

    // treeHTML :: tree
    //      {tag :: String, text :: String, kvs :: Dict}
    //      -> String
    const treeHTML = tree =>
        foldTree(
            (x, xs) => `<${x.tag + attribString(x.kvs)}>` + (
                'text' in x ? (
                    x.text
                ) : '\n'
            ) + concat(xs) + `</${x.tag}>\n`)(
            tree
        );

    // attribString :: Dict -> String
    const attribString = dct =>
        dct ? (
            ' ' + Object.keys(dct)
            .reduce(
                (a, k) => a + k + '="' + dct[k] + '" ', ''
            ).trim()
        ) : '';

    // TEST ---------------------------------------------
    const main = () => {
        const
            tableStyle = {
                style: "width:25%; border:2px solid silver;"
            },
            trStyle = {
                style: "border:1px solid silver;text-align:right;"
            },
            strCaption = 'Table generated by JS';

        const
            n = 3,
            colNames = take(n)(enumFrom('A')),
            dataRows = map(
                x => Tuple(x)(map(randomRInt(100)(9999))(
                    colNames
                )))(take(n)(enumFrom(1)));

        const
            // TABLE AS TREE STRUCTURE  -----------------
            tableTree = Node({
                    tag: 'table',
                    kvs: tableStyle
                },
                append([
                    Node({
                        tag: 'caption',
                        text: 'Table source generated by JS'
                    }),
                    // HEADER ROW -----------------------
                    Node({
                            tag: 'tr',
                        },
                        map(k => Node({
                            tag: 'th',
                            kvs: {
                                style: "text-align:right;"
                            },
                            text: k
                        }))(cons('')(colNames))
                    )
                    // DATA ROWS ------------------------
                ])(map(tpl => Node({
                    tag: 'tr',
                    kvs: trStyle
                }, cons(
                    Node({
                        tag: 'th',
                        text: fst(tpl)
                    }))(
                    map(v => Node({
                        tag: 'td',
                        text: v.toString()
                    }))(snd(tpl))
                )))(dataRows))
            );

        // Return a value and/or apply console.log to it.
        // (JS embeddings vary in their IO channels)
        const strHTML = treeHTML(tableTree);
        return (
            console.log(strHTML)
            //strHTML
        );
    };


    // GENERIC FUNCTIONS --------------------------------

    // Node :: a -> [Tree a] -> Tree a
    const Node = (v, xs) => ({
        type: 'Node',
        root: v,
        nest: xs || []
    });

    // Tuple (,) :: a -> b -> (a, b)
    const Tuple = a => b => ({
        type: 'Tuple',
        '0': a,
        '1': b,
        length: 2
    });

    // append (++) :: [a] -> [a] -> [a]
    // append (++) :: String -> String -> String
    const append = xs => ys => xs.concat(ys);

    // chr :: Int -> Char
    const chr = String.fromCodePoint;

    // concat :: [[a]] -> [a]
    // concat :: [String] -> String
    const concat = xs =>
        0 < xs.length ? (() => {
            const unit = 'string' !== typeof xs[0] ? (
                []
            ) : '';
            return unit.concat.apply(unit, xs);
        })() : [];

    // cons :: a -> [a] -> [a]
    const cons = x => xs => [x].concat(xs);

    // enumFrom :: a -> [a]
    function* enumFrom(x) {
        let v = x;
        while (true) {
            yield v;
            v = succ(v);
        }
    }

    // enumFromToChar :: Char -> Char -> [Char]
    const enumFromToChar = m => n => {
        const [intM, intN] = [m, n].map(
            x => x.charCodeAt(0)
        );
        return Array.from({
            length: Math.floor(intN - intM) + 1
        }, (_, i) => String.fromCodePoint(intM + i));
    };

    // foldTree :: (a -> [b] -> b) -> Tree a -> b
    const foldTree = f => tree => {
        const go = node =>
            f(node.root, node.nest.map(go));
        return go(tree);
    };

    // fst :: (a, b) -> a
    const fst = tpl => tpl[0];

    // isChar :: a -> Bool
    const isChar = x =>
        ('string' === typeof x) && (1 === x.length);

    // map :: (a -> b) -> [a] -> [b]
    const map = f => xs =>
        (Array.isArray(xs) ? (
            xs
        ) : xs.split('')).map(f);

    // ord :: Char -> Int
    const ord = c => c.codePointAt(0);

    // randomRInt :: Int -> Int -> () -> Int
    const randomRInt = low => high => () =>
        low + Math.floor(
            (Math.random() * ((high - low) + 1))
        );

    // snd :: (a, b) -> b
    const snd = tpl => tpl[1];

    // succ :: Enum a => a -> a
    const succ = x =>
        isChar(x) ? (
            chr(1 + ord(x))
        ) : isNaN(x) ? (
            undefined
        ) : 1 + x;

    // take :: Int -> [a] -> [a]
    // take :: Int -> String -> String
    const take = n => xs =>
        'GeneratorFunction' !== xs.constructor.constructor.name ? (
            xs.slice(0, n)
        ) : [].concat.apply([], Array.from({
            length: n
        }, () => {
            const x = xs.next();
            return x.done ? [] : [x.value];
        }));

    // MAIN ---
    return main();
})();
