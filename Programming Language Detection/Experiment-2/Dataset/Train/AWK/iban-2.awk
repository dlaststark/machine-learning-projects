cat > test.iban
FR33 ^__^ 0BAD
AA11 1234 6543 1212
FR33 1234 5432
CH93 0076 2011      6238 5295 7
GB82 WEST 1234 5698 7654 32
GB82 TEST 1234 5698 7654 32
^D
gawk -Mf iban.gawk test.iban
