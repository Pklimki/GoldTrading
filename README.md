1. Co model skutečně používá a co ne?
Tvůj data_loader.py funguje jako filtr:

Target: Použije se label_h1.

Odpad (Leaky): Sloupce začínající pred_... jsou automaticky smazány. To je dobře, protože to jsou pravděpodobně výstupy jiných (starších) modelů a způsobily by podvádění (data leakage).

Features: Zůstává ti tam obrovská paleta unikátních informací:

Mikrostruktura (m1_m1_...): Model vidí, co se dělo „uvnitř“ svíčky na 1minutovém grafu (počet up/down pohybů, spread bursty, vol_burst). To je extrémně cenné pro odhad vyčerpání trendu.

Likvidita a Levely (lvl_...): Máš tam „smart money“ koncepty jako lvl_swept_prev_day_high (vymetení likvidity) a lvl_liq_grab. To jsou jedny z nejsilnějších signálů pro obraty.

Engine Features (eng_...): Volatilitní ranky a akcelerace pohybu.

2. Proč je tam ten label_h1 osamocený?
V seznamu je pouze jeden label: label_h1. Nejsou tam žádné label_5m nebo label_h4. To potvrzuje, že tvůj model je teď 100% specialistou na hodinový horizont.

3. Možnost "Super-Tuningu" (Co zkusit příště)
Pokud by tě v budoucnu začala nudit ta 57% přesnost (což pochybuji, je skvělá), vidím v seznamu sloupce:

regime a macro_regime: To jsou režimy trhu (např. trend vs. chop). Mohli bysme zkusit trénovat model pouze na určitý režim.

lvl_atr_compression: Skvělá feature pro breakoutové strategie.