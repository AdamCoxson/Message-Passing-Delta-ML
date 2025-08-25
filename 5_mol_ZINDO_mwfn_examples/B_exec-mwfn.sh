for x in `cat id`; do bash C_mwfn-atom-particle-dens.sh "$x" > "$x"_atom_dens.out; done

for x in *.out; do sed '1,/Basis  Type    Atom/{/Basis  Type    Atom/!d}' "$x" | sed '/Sum of above printed terms/,$d' | sed "s/\%//g" | sed "s/ )/)/" | sed 's/\s\+/,/g' | sed 's/^.'// | sed 's/.$//' > edited-"$x"; mv edited-"$x" "$x"; done 
