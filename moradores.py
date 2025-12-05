import pandas as pd
import re
import io

# Raw OCR text from the user's prompt (concatenated)
ocr_text = """
0101 1 Erika Funari Ossucci Proprietário 05109433909 erikaossucci@hotmail.com
0102 1 Maria Eduarda da Silva Souza Proprietário 38592960860 maria_eduarda_souza_@hotmail.com
0103 1 Ana Paula Azair dos Santos Proprietário 02763902960 paulynhaapis@hotmail.com
Nicoli Caires Almeida
(44) 99174-5643
Residente nicolicaires@outlook.com;marlisugawara@gmail.com
0104 1 Thiago Augusto Dias
(44)3255-1407 ((44)99815-4600)
Proprietário 04969190917
Fabio Teodoro de Souza
(44) 99937-4161
Residente
0105 1 Elisabete Gonçalves Simoni
(44) 3027-8109
Proprietário 49544861904 bete_simoni@hotmail.com
0106 1 Fernanda Regina Marcolini
(44) 99835-3705
Residente fernanda_frm92@hotmail.com
Fernanda Regina Marcolini Proprietário fernanda_frm92@hotmail.com
0107 1 Adriana Carla Gomes
(44) 3034-8080
Proprietário 51074966520 adriana_familia@hotmail.com
0108 1 Thiago Vinicius Okada
(44) 3226-2587
Proprietário 07089262901 thiago.okada@hotmail.com
Lilian Stephane de Freitas Residente 05893656903 thiago.okada@hotmail.com
0201 1 Alex Sandro Cercunvis
(44) 3263-6269; (44) 3224-1239
Proprietário 02067856901 alex@ossucci.com.br
0202 1 Jonas Alves de Souza Junior
(44) 3023-4379
Proprietário 04961263931 jjr.86@hotmail.com
0203 1 Vanice Maria Juliani
(44) 3263-3743
Proprietário 50130307904 meljdias@hotmail.com
0204 1 Rogerio Machado Silva
(44) 3025-3067 ((44)998936792;998322209)
Proprietário 14791511816 mmarinelo@yahoo.com
0205 1 Neide Aparecida Mavhado
(44) 3275-1144
Proprietário 76650073920 juliomachado_39@hotmail.com
Júlio Cezar Rodrigues Machado
(44)998953458
Residente
0206 1 Jamil Abd Mahmund Saleh Awadallak Proprietário 50779141920 jamilbd@bol.com.br
Edimery de Souza Barcelosi
(44)99133-2403
Residente
0207 1 Pedro Gonçalves
(44)33546067
Proprietário 39712036987 sargento_gonca@hotmail.com
0208 1 Edson José Gomes
(44)30114889
Proprietário 74835416953 gomesedson2@hotmail.com.br
Andrá Payão Aguilera
(44)33616085
Residente 40141395818 andre_payao@hotmail.com
0301 1 Alexandre Sassi de Brito
(44)33054972
Proprietário 00864901925 odetesassi@hotmail.com
0302 1 Erica Silva Fontana
(44)30349208(44)32248091
Proprietário 05257209950 fontanaerica@hotmail.com
0303 1 Douglas Lopes Faria
(44)32556053
Proprietário 00642766940 douglaslopesf@gmail.com
0304 1 Patricia Andyara Thibes
(44)30348880 ((44)999046028;999529761)
Proprietário 01015018971 patiandyara@hotmail.com;ccarpena@adapar.pr.gov.br
Claudia Aparecida Nascimento Moreira
+55 18 996737749
Residente 09765152884 fabigilioli81@gmail.com
0305 1 Claudemir José Trindade
+55 43991553175
Proprietário 86344170930 claudemirjosetrindade64@gmail.com;
valeria3trindade@gmail.com
0306 1 Solange Aparecida Teixeira Proprietário 06112796902 sol.teixeiramga@hotmail.com
0307 1 Antônio Marcos Barbeta
(44)32287541
Proprietário 07123081994 antonio.barbeta@hotmail.com
Milena Kecya Guerra
+55 43999880258
Residente 09333925910 milena_kecya@hotmail.com
0308 1 Maria Aparecida Farias Santos
(44)32253555 ((44)998847036)
Proprietário 58800158587 garcia.luiza@hotmail.com;gaabiifarias@hotmail.com
0401 1 Regina Alves Thon
(44)30283654 ((44)988116686)
Proprietário 03288849950 registhon@hotmail.com;reginathon@unipar.br
Francisco Carlos da Silva
(67)981517047
Residente
0402 1 Daniel Frederico Mayer
(44)30238359
Proprietário 93036205934 df_mayer@hotmail.com
Cristiano Trzanski Terlan
(54)997155953
Residente 03225277088 arineprinz@hotmail.com;cristianotrzcinski@outlook.com.br
0403 1 Noel Pedro de Sousa
(44)36352182 ((44)999512830)
Proprietário 32384190997
Alfredo Shigueru Takano
+55 44988011917
Residente 57784825900 alfredotakano24@gmail.com
0404 1 Luiz Gustavo Martins Ferreira
(44)30266377
Proprietário 04010454989 gustavo.edf@outlook.com
0405 1 Valdir Marques
(44)32621648 ((44)991139612;997631030)
Proprietário 49623303904
0406 1 Vinicio Noda
(44)30232941 ((44)999619940)
Proprietário 06886105957 vinicionoda@hotmail.com;vinicionoda@gmail.com;
anne_caroline0102@hotmail.com
0407 1 Robson Cleiton de Souza
(44)32662635
Proprietário 00928516903 rob_souza@hotmail.com;fabiana.ferracioli01@gmail.com
0408 1 Jhonatan de Carvalho dos Santos
(13)98821712
Proprietário 38982301801 jcarvalho26@hotmail.com
0501 1 Franciele Alves da Silva Schuerman
(44)33540960 ((44)997634237;999727335)
Proprietário 04221815957 rafael_schuermann@hotmail.com;franciele.alves.
silva@outlook.com
0502 1 Cleumira Dias
(44)988115402
Proprietário 54968984987 cepietrobon@hotmail.com;cleoghiraldi@hotmail.com
0503 1 Paulo Alberto Bernardino
(44)32441148 ((44)999115625)
Proprietário 05267824984 paulo1908@gmail.com
0504 1 Nilson Tadashi Uhemura
(44)33015183 ((44)999020303)
Proprietário 85667838753 adm@institutodeolhos-maringa.com.br
Pollianny Cristiny Monteiro
+55 44988441961;+55 44998352448
Residente 12100623990 kinukawa17@gmail.com
0505 1 Rodrigo Aparecido Roque
(44)32761275
Proprietário 05846698956 rodrigo@ghelere.com.br
0506 1 Ina Maria Frias Cabral
(44)30289755 ((44)999128636)
Proprietário 36577596972 ina@datadoctor.com.br
Matilde Lucia S. Gomes
(44)998320043
Residente 55076742991 gedsonlsg@gmail.com
0507 1 Demilson Amaral
(44)33057059
Proprietário 38741067991 demilsonamaral2@gmail.com
0508 1 William Kenji Umeki
+55 44991512173
Residente 49846785852 kenjiumk@gmail.com
Nicolas Renan Valim Bonani
+55 44998595879
Proprietário 12041612916 nicolas_bonani@hotmail.com
0601 1 Silvana Bastos Pinto
(44)32261940
Proprietário 03280456916 silvanabastos_@hotmail.com
0602 1 Marcel Henrique Goulart
(44)32233151(44)30261124
Proprietário 07644489940 marcelhgoulart@gmail.com
0603 1 Marcia Aparecida Reis
(44)30236872
Proprietário 05387096935 marciareis_mm@hotmail.com
0604 1 Vergílio Vitorino Bernardino
32640106
Proprietário 38784041900
Lorenah Leão Floro da Silva Goes
+55 44997370813
Residente 06151603958 Lorenah_leao@hotmail.com
0605 1 Luciana Galdino Pereira
998500671
Proprietário 82199345491 pajelu@msn.com
Juliety Christine Galdino Alves
+55 44988222133
Residente 07111559924 julietyalves.psi@gmail.com
0606 1 Isabel da Silva Dantas Bonacin
33467467 (991680603;99736396)
Proprietário 67726917920 isdbonacin@hotmail.com
0607 1 Vergilio Vitoria Bernardino
32620106
Proprietário 38784041900
Lucas Piccolo de Oliveira Residente 09141780914 lucas_piccolo@hotmail.com
(44)999097970
0608 1 Otair Beloto Proprietário 78424879953 otabeloto@gmail.com
0701 1 Deborah Graciano Martin
3226-2444 (991266163;(54)991069231)
Proprietário 04378579995 dgmmga@hotmail.com
Gregory Vermolhen de Barros
+55 449988808434
Residente 11809911923 gregory40barros@gmail.com
0702 1 Douglas Oliveira Moretti
32252467 (999785330)
Proprietário 00781214939 douglas@coopercred.com;douglas.moretti@coopercob.
com.br
0703 1 Luis Andres Otero
33058586 (99884448)
Proprietário 05075984908 tino.otero@gmail.com
Lucas Piccolo de Oliveira
44999097970
Residente 09141780914 lucas_piccolo@hotmail.com
0704 1 NilsonTadashi Uhemura
33015183 (999020303)
Proprietário 85667838753 adm@institutodeolhos-maringa.com.br
0705 1 Ednilson Cicerce
(44)41415722
Proprietário 02523577950
0706 1 Natalina Vanilde Botam Proprietário 14760064826
David Lapa de Cerqueira Junior
+55 21 991615572;+55 21 982225095
Residente 79188800725 d.lapa@ig.com.br;maria.sonia77@hotmail.com
0707 1 Helena Conceição dos Santos Barbosa
(44)999332381
Proprietário 49010891968 helena.barbosa1965@hotmail.com
0708 1 Irani Guillaumon Pereira da Silva
(11)23345459 ((11)945426451;945426451)
Proprietário 27842782806 iranig@terra.com.br;raissa.andrade@hotmail.com;
iranisil@gmail.com
Melissa Manuelle Ferraz
(44)997132136
Residente 13281359930 melissaferraz02@gmail.com
0801 1 Alexandre Moran Junior Proprietário 11457904969 alexandremoran@gmail.com;locacao2@gcli.com.br
0802 1 Leandro Corbelo
(44)30235701
Proprietário 04101761922 leandroxk@gmail.com
0803 1 Valdirene Aparecida Brito Sandes
+55 66 84484981;+55 66 9960 47602
Proprietário 95880615987 financeiro@eletroara.com;joaopedrosandes00@gmail.com
0804 1 Marilene Aparecida Mulatti Costa
+55 44988012520
Proprietário 71373063904 marilenemulatti@hotmail.com
0805 1 Fabricia Rosa Rubini
(44) 32656222
Proprietário 03202337904 fabriciarubini@gmail.com
Estela Voltarelli de Cesare
(44)33464403 ((18)981794600)
Residente estela@especialvet.com.br;jovirasi@gmail.com;estela.
decesare@gmail.com
0806 1 PLANALTO ADMINISTRADORA DE BENS LTDA
(44)30284777 ((44)99677222)
Proprietário 13234106000100
Carla Simone Bernanardo Musiau
(43)34651132 ((43)984096100)
Residente
0807 1 João Avelino da Silva
(44)32381447 ((44)988057002;98447619)
Proprietário 44530943968
Edson Cezar Inácio Residente 92907288920 e_c_inacio@hotmail.com
(44)998731412
0808 1 Claudio Emanuel Pietrobon
998206291;998773500;998463839
Proprietário 45418624849 bianca.agosto@outlook.com;l-baggio-mlro@hotmail.com;
cepietrobon@hotmail.com;izapazinato@hotmail.com
0101 2 Gabriele Crispim Rigole
32656921 (32645157)
Proprietário 01035852900 gabicr2004@yahoo.com.br
0102 2 Rodrigo Malaquias Costa
30263834
Proprietário 08133419905 rodrygo_mc@hotmail.com
Celso de Marchi
998129343 (999324648)
Residente
0103 2 Kleber da Costa
32670023
Proprietário 05317698952 kleberdacosta85@hotmail.com
0104 2 Terezinha Baraneka
(44)999109092
Proprietário 44983042920 terezinha.baraneka@gmail.com
0105 2 Cristiane Morente
30296505 (32218518)
Proprietário 04108653971 cristiane.morente@hotmail.com
Guilherme Augusto Cruz
(44)991124697
Residente 40397498802 gui.cruz94@gmail.com
0106 2 Jackson da Silva Bezerra
33018365
Proprietário 06774872905 jackson_dasilva@hotmail.com;jacksonbezerra1989@gmail.
com
0107 2 Leonardo José Romanini
32618136 (98986005)
Proprietário 03847679902 ljromanini@gmail.com
Camila Maria da Silva
+55 16993328961;+55 16981191795
Residente 22766463810 camilaiaia@bol.com.br;robertafigueiredo73@gmail.com
0108 2 Jayson Seiji Takeda
997219862 (998489577)
Proprietário 05275414943 jayson_takeda@hotmail.com
0201 2 Rafael Di Domenico
30282007 (999100303)
Proprietário 00878903992 grilo_bike@hotmail.com
Magali Toledo
999350252
Residente
0202 2 Bianca Franciele Hanzen
+55 44991367536
Proprietário 06967050958 biancahanzen@gmail.com
0203 2 Juliano de Lima
32223222
Proprietário 05588506936 jlndlima85@gmail.com
0204 2 Armando Barreto
32657542 (99059095)
Proprietário 48539562987 vanessalaila_barreto@hotmail.com
Anderson Kazuo Taniuchi
998773500 (998206291)
Residente anderkaz70@yahoo.com
0205 2 Claudio Emanuel Pietrobon
998206291 (998773500)
Proprietário 45418624849 bianca.agosto@outlook.com;l-baggio-mlro@hotmail.com;
cepietrobon@hotmail.com;izapazinato@hotmail.com
0206 2 Emanuel Flavio Kloster
30287119 (998236689)
Proprietário 02639776924 financeiro@suprimedimplantes.com;
emanuel@suprimedimplantes.com
0207 2 Paula Fidelis dos Santos
30293813
Proprietário 00764232983 parati_modas@hotmail.com
Elizabete Gloria Faker
999236968
Residente betefaker71@hotmail.com
0208 2 Katia Nascimento Aguilar Lopes
44999072809
Proprietário 03594673936 ambulancia@santaritasaude.com.br;katia.aguilar.lopes.
ka@hotmail.com
0301 2 Thiago Paulino da Silva Bibiano
4432285903
Proprietário 05419574900 odvan_thiago@hotmail.com;odvan544@gmail.com
Alan David Conte
4433019362-4432251060 (4499152901)
Residente 04805803916
0302 2 Alan David Conte
4433019362-32251060 (4499152901)
Proprietário 04805803916
Wellington Keity Ueda
44999064006
Residente ueda_marcelo@hotmail.com
0303 2 PREVER ANGELUS ADMINISTRADORA LTDA.
44988122831
Proprietário 33149334000180 boris.oliveira@grupopreversul.com.br
CARLA MARTINS DE SOUZA SALVALAGIO
+55 44997568741
Residente 09262613989 carlamartins07souza@hotmail.com
0304 2 Lucas da Silva Mota
32258623-33540176
Proprietário 05468935970 vanessa_generali@hotmail.com;silvalucas00231@gmail.
com;maycon.delarme22@gmail.com
0305 2 PLANALTO ADMINISTRADORA DE BENS LTDA
44-30284777 (4499677222;4499671891)
Proprietário 13234106000100
0306 2 Everson Luiz Turra
44-30244451 (4499339708)
Proprietário 51728419972 eversonturra@gmail.com;eversonturra@bol.com.br
Geny Costa dos Santos
+55 44998337170;+55 44999824142
Residente 01776733975 simone.costa3@hotmail.com
0307 2 Ricardo Montanher
44 99914-5278
Proprietário 00501346988 ricardomontanher81@gmail.com;ailton.
marquesjunior2016@outlook.com
Guilherme Santiago Mendes Cantalice
+55 81991054339;+55 81994012924
Residente 08373980458 guigacantalice@gmail.com;aragaorisia18@gmail.com
0308 2 Antônio Claudemir de Melo Proprietário 70193053934 claudemirmelo@hotmail.com
0401 2 Adriano Hideo Taguchi
(44)30333444 ((44)991592900)
Proprietário
0402 2 Artur Faccin De Souza
(44)32501758
Proprietário 04961831921 tuifs@hotmail.com
0403 2 Lucas Henrique Maldonado Da Silva
(44)999019585
Proprietário 07569359976 lucasmaldonado7@gmail.com
0404 2 Adriana Paula Alves Ferreira
(44)3245-4861
Proprietário 04233710969 adrianapa.alves@gmail.com
0405 2 Genilson Felix Da Silva
(44)3005-2036 ((44)995867695)
Proprietário 81148461949
Ivone José De Souza Da Silva
(44)99823-5697
Residente ivonefelix44@gmail.com
0406 2 Antonio Sardoneli Proprietário 23497220906 sardonelli-a@hotmail.com;solangeboning@hotmail.com
Rosangela Teixeira Brant Residente 05440997962 ro_brant@hotmail.com
44-999596425
0407 2 Fernanda Karolline Tobias
(44) 3259-2801 3020-2438 ((44)984368010;991428569)
Proprietário 07001334960 fernanda_tobias@hotmail.com;gustavo@nevan.com.br
0408 2 Wesley da Costa Guimarães
11971548524
Proprietário 35166639871 weslei_guimaraes@hotmail.com
0501 2 Rafaela Da Rocha Carreira
(44)3226-3441 997112029 ((44)998283997;999473220)
Proprietário cleucialopes@gmail.com;rafaelacarreira@hotmail.com;
renatoalberto84@gmail.com;marcella.2288@gmail.com
Maria Aparecida Soares do Amorim
+55 44998565514
Residente 89957164953 tidamorim@hotmail.com
0502 2 Fernando Borsalli Verderio
(44) 3227-6231 ((44)99902911)
Proprietário 02987871996 fernando.verder@gmail.com;werner.phw@gmail.com
Nailson De Souza
(44)3026-8473 ((44)988448543)
Residente nailson@embalansburiti.com.br
0503 2 Paulo Rogerio Ayme Da Silva
(44)3218-8600
Proprietário 05716821979 pauloayme@hotmail.com
Roseli Borgan Guandalino dos Santos
+55 44997078206
Residente 78460425991 roseliborgan767@gmail.com
0504 2 MRV ENGENHARIA E PARTICIPACOES SA Proprietário 08343492000120
0505 2 Claudio Emanuel Pietrobon
998463839 ((44)998206291;998773500)
Proprietário 45418624849 bianca.agosto@outlook.com;l-baggio-mlro@hotmail.com;
cepietro@hotmail.com;izapazinato@hotmail.com
Clara Izabela Tieppo
(44)3247-1519 ((44)99703-2004)
Residente izabelatieppo@gmail.com
0506 2 Claudemir Alves de Oliveira
(44)998900741
Proprietário weskwyortiz16@gmail.com
Lavinia Nicola Salvador
+55 18997701508
Residente 44911871857 laviniasnicola@gmail.com
0507 2 Fernando Jorgeto Da Silva
(44) 3225-0144 ((44)998561306)
Proprietário 06339111947 fernando_supergasbras@hotmail.com;fernandojorgeto.
adv@gmail.com
Pedro Henrique Ribeiro Vicente de Souza
(41)99799-1248 ((44)99870-9160)
Residente 11913186997 phvicente@hotmail.com
0508 2 Davi Frederico Kruse
(44)33050220
Proprietário 07058418985 davi_kruse@hotmail.com
Fernanda Lourdes Klauck
+55 44988615005
Residente 11320403964 fernanda_klauck@hotmail.com
0601 2 Daniela Peres
4433056596 (4499876596)
Proprietário 04688069979 peresdanni@hotmail.com
Thiago Bitencourt
44991372888
Residente bitencourt409@gmail.com
0602 2 Mauricio Akira Tanaka
4433461446
Proprietário 50041258991 mauricioakiratanaka@hotmail.com
0603 2 Alan Belini
4432226122 (4430313015)
Proprietário 03831659974 alan6225@hotmail.com
0604 2 Antonio Soares da Silva Neto Proprietário 44440219934
4432654316
0605 2 Fernanda Maria Moura
4433419654 (4497036270)
Proprietário 07637184919 fernandadondoka@hotmail.com
Edervan Barbosa Gardini
44998430728
Residente 08088271975 edervan_bg@hotmail.com
0606 2 Maria Eduarda Carniatto
4432252930 (4499766220)
Proprietário 04171877938 nati_carniatto@hotmail.com
Vilson Flavio de Andrade Junior
+55 44998759418
Residente 11635580951 vilson.flavio.junior@gmail.com
0607 2 Bruna Corteline Morimoto
4433019122 (4499733219)
Proprietário 05596250902 cobranca@leloimoveis.com.br;bruna.morimoto@yahoo.
com.br;marlene_morimoto@yahoo.com.br
Lelo Imoveis Residente
0608 2 Leila Cristina Vicente Lopes
4432244386 (4499514264)
Proprietário 60786655968 leilla.lopes@bellinatiperez.com.br;paresdanni@hotmail.com
Matilde de Fatima Tertuliano
4499987659 (44999201244)
Residente 02122168994 peresdanii@hotmail.com
0701 2 Maria Margarida Cavalli
4498762574
Proprietário 94604452920
0702 2 Pedro Henrique Iwamoto
+55 44988224989
Proprietário 06439573918 phiwamoto@hotmail.com
0703 2 Tiago Fernando Bocardi da Silva
4432683483 (4499700095)
Proprietário 05265309950 fbs.tiago@gmail.com
Eliana Correa
4497456412 (4497456412)
Residente
0704 2 Cleverson Nascimento de Mello
(44) 99844-5474 
Proprietário 08851214905 cleversonmello@gmail.com
0705 2 Gustavo Gonçalves Lourenço
4432501224 (4484193222)
Proprietário 04961832901 supermercadosantaluzia@gmail.com;
esteticaivanabertoli@hotmail.com
Ademir Residente
0706 2 Marcelo Saraiva Muniz
4432670765 (44988049428;4499454682)
Proprietário 03865243940 marcelosaraiva84@outook.com;
bigfrios_distribuidora@hotmail.com;cobranca@leloimoveis.
com.br
Lelo Imoveis Residente
0707 2 Bruno Piva
4432642955 (44998183400)
Proprietário 00932737978 eliane_casellato@hotmail.com
0708 2 Margareth Alves dos Santos
44999591208
Proprietário 85724130904 margozacaria@hotmail.com
Lucas da Silva Barbosa
24999011139 (24999011133)
Residente 12019580756 lucasbarborsa1987@outlook.com
0801 2 Diego Franco Pereira
4433056991 (4499505441)
Proprietário 00972528911 diegofrancopereira@hotmail.com
0802 2 Bianca da Rocha Pietrobon
998773500 (46999730073)
Proprietário 00828848971 bianca.agosto@outlook.com;l-baggio-mlro@hotmail.com
Lorena Wolfart Pazin
(0xx44)997703917
Residente 10704989980 lorenalorenawolfart@gmail.com
0803 2 Genivaldo Bono
4430281429 (44984329622)
Proprietário 74846329968 thbono@gmail.com
Carlos Alexandre de Paula
44999212549 (44999212549)
Residente 06655685923 carlos_alexandr3@hotmail.com
0804 2 MARCOS VERENHITACH Proprietário 07811711990 marcosvereni@gmail.com
ROBERTH PITARRO DE SOUZA SANTOLAIA MARTINS
+55 44997032069;+55 44998335754
Residente 10565083961 roberthsanto20@gmail.com;beatrizgoularm@gmail.com
0805 2 Daniel Tadeu Sanches Xavier
4433544445
Proprietário 05256794990 danieltsx@hotmail.com
0806 2 Marcos da Silva
4432744403 (4497104965)
Proprietário 08506982944 marcossilva172@hotmail.com
Neuza
44988384834
Residente
0807 2 Therezinha Balduino Cesso
4430370090 (44997301340)
Proprietário 43381898949 terezinhaprmj@gmail.com
0808 2 Cleuza Martins Caetano
4430290923 (4499310007;4499500303;4499760983)
Proprietário 95918710906 kracheski55@hotmail.com
0101 3 Analia Ferreira Duarte
44991099859
Proprietário 56992840100 analia.47@hotmail.com;pauladuarte0@hotmail.com
0102 3 Alexsandro Aparecido da Silva
4430287178 (44998721002)
Proprietário 13818283885 alex.mga@bol.com.br
0103 3 Thiago de Oliveira Santos
4432223232 (44999327912)
Proprietário 05925388933 gisele1_bueno@hotmail.com
0104 3 Alan Jonata Ribas
4498570574
Proprietário 01005514917 alan_jonata@hotmail.com
0105 3 Vitória Cortes Caleffi
+55 44991010612
Proprietário 10154673935 leffsleffs@gmail.com
0106 3 Iramir Eugenio Silva
4430285037 (44984072604)
Proprietário 02554885990 iramir2008@hotmail.com
0107 3 Marcos Makoto Yamada
4432323232 (4499278339)
Proprietário 85149047953
0108 3 Josimeia Aparecida Ferreira
4430314383 (4499118947)
Proprietário 48093874968 josimeia_ferreira@hotmail.com
0201 3 Luciano Pereira dos Santos
4432463920 (4499271099)
Proprietário 05181014950 lucianopereira1985@hotmail.com
0202 3 Carolina Alvas Falavina
4497033466
Proprietário 05774443957 carolfalavina@hotmail.com
0203 3 Anderson Marcos Correia
4433019770 (44991346664)
Proprietário 02984825926 correia@portosecoparana.com.br;anderson.
marcosmga@gmail.com
0204 3 Vanilda Lemos de Oliveira
4499708500
Proprietário 02368796851 elemosoliveira@yahoo.com.br
0205 3 Christian Gelati Proprietário 08453715920
Rozeli Menegazzo
+55 43999806796
Residente 50489240968 taimenegazzo2@gmail.com
0206 3 Jonathan Alves Rodrigues
+55 44991357172
Residente 06433223613 jonathanar@gmail.com
Melissa da Silva Oliveira Honorio
+55 18997746807
Proprietário 41860204856 melissa.honorio123@icloud.com
0207 3 Vanessa Cristina Teixeira Goes
4499333325 (4499333325)
Proprietário 05629541986 wanessagoes@hotmail.com
0208 3 Oswaldo Magi Filho
4432781591 (44988319123)
Proprietário 02925308970 oswaldo.magi@gmail.com
Rosimeire Lopes Langaro
+55 44988112529;+55 43933002516
Residente 04051704912 daniiielrodrigues@gmail.com;meirelangaro9@gmail.com
0301 3 Silvania Lopes Ferreira
4432275000 (4491029303)
Proprietário 01813723982 jhonathan.dto@hotmail.com
0302 3 Simone Nochelli
44997651026 (44999162676)
Proprietário 05198472964 simone_nochelli@hotmail.com
0303 3 Aldemir Borges dos Santos
4432687309 (4432241975)
Proprietário 71699074968 ademiresandra@gmail.com
0304 3 Jorge Alison de Souza
44998982395
Proprietário 05046001926 jorgemeb25@outlook.com;jorgemeb1212@outlook.com
0305 3 Jessika Nabao Lopes Ferreira
44998568650
Proprietário 07196324986 jessika_nl@hotmail.com
0306 3 Valdir De Oliveira
(44)3268-8198 3011-4302
Proprietário 63451875934 valdirmga@yahoo.com
0307 3 Rafael Luiz Neves Amaro
(44)3218-6300 ((44)99743124)
Proprietário 05894297958 rafael.treezy@gmail.com;marin.amaro@hotmail.com
0308 3 Amanda Da Silva Vidal
(44)3268-1442 3026-4666
Proprietário 05957636932 amandda_89@hotmail.com
0401 3 Elizete De Fátima Botan
(44)997210095
Proprietário 60208961968
0402 3 Jefferson Hideki Tookuni
4432431132 (44988241741)
Proprietário 08053500986 jefferson@idbrasil.com
0403 3 Rodrigo Fernandes Novo
4432598919
Proprietário 18554137876 rodrigonovo@hotmail.com
0404 3 Ellen Jaqueline da Silva
4432662824 (44998126322;67993299055)
Proprietário 06442959961 ellen_lak@hotmail.com
TAINA SILVA BIGI
+55 79991158588
Residente 06643323584 tainabigi@hotmail.com
0405 3 Klabyr Wanderson Cristovao de Jesus
4130199773 (32259305)
Proprietário 02534257943 locacao.ldsmoveis@gmail.com;ldaimoveis@gmail.com;
klabyr@hotmail.com;gislaine@maxbelt.com.br
Daiane Medeiros
+55 43998189089
Residente 08518390966 daiane_453@hotmail.com
0406 3 Jefferson Tiago Correia do Prado
32622195
Proprietário 01045351997 jeffersonprado18@gmail.com
Yuri Wanderley Hahne Residente 28667292850 yurihahne@hotmail.com
0407 3 Andressa Lopes Amaral
4430346649 (44999222143)
Proprietário 06435975906 andressalopesamaral@hotmail.com;gilschooner@gmail.
com
0408 3 Custodio Ribeiro Vieira
4432671101 (4499111246)
Proprietário 47676710944 custodiovieira@ig.com.br;custodioribeirovieira@gmail.com
0501 3 Vinicius Dias Paes
4432553121 (4499733891)
Proprietário 00969378904 vinicius@ectom.com.br
0502 3 Carlos Gomes da Silva
4499977773
Proprietário 30304365866 carlosfarmaciagomes@hotmail.com
0503 3 Anderson Ferreira Bernardo
4430463201
Proprietário 26838862840 bernardo_775@hotmail.com
Allisson Gabriel Idelfonso Bistaffa Residente 08945869980
0504 3 Joanes de Castro Junior
4432250431
Proprietário 05269078945 joanescastro@hotmail.com
0505 3 Mariangela Costa Luz
4430346930
Proprietário 05324782823 ferrossibergamo@gmail.com
0506 3 Julio Cezar Florencio da Cunha
4430259291
Proprietário 04974815962 julio_jcc@hotmail.com
Jessica Carolina Crepaldi Costa
44998313242
Residente 06644603998 jessica_crepaldi_@hotmail.com
0507 3 Rodrigo Aparecido Rodrigues Moretti
44 9708-6497
Proprietário 71792554168 rodrigo-moretti@hotmail.com
0508 3 Renan Alves Batista
4432681645
Proprietário 06306287930 renanalvesb@gmail.com
Rubens Borges de Medeiros
+55 43999126290;+55 42996507881
Residente 17416485991 anabia_corsini@hotmail.com;gabrielypereira230@gmail.
com
0601 3 Adriana da Ros
4432591180 (4430462600)
Proprietário 02543237909 tesouraria@grupoagrolatina.com.br
Erick Matheus Veloso da Cruz
44999716415 (44999716415)
Residente 08946381906 erik999matheus@gmail.com
0602 3 Jhonatan de Carvalho dos Santos
44984115219
Proprietário 38982301801 jhonatan_fieltorcida@outlook.com
0603 3 Teresa Cristrina Mendes
4432449232
Proprietário 81584199920 crismendes3@hotmail.com
0604 3 Joao Anderson Carbo
4432467072
Proprietário 03189048967
0605 3 Geisa Lainara Almendro
(44)998724729
Proprietário 08607752910 geisalainara@gmail.com
0606 3 Desiree Cardoso Caramori
4432267282
Proprietário 07295335900 desireewish@hotmail.com
0607 3 Doroty Vieira Zanelato Proprietário 55613330972 damaziozanelatojr@live.com
4434631589 (44999740516)
0608 3 Aline Patricia Carabinoski da Silva
4432277144
Proprietário 07938308935 aline.patricia2011@bol.com.br
0701 3 Andre Stefani
4432658122 (44999196540)
Proprietário 03917268973 sttheffany@hotmail.com
0702 3 Cristiane Regina Guilherme
4499728073335
Proprietário 76070905920 cris_thanise@yahoo.com.br;crisguilherme@bb.com.br
Thiago Andrian do Amaral Residente 10067312969 kattlleya.andrian@escola.pr.gov.br
0703 3 Mauro Cesar Adame
4433059985
Proprietário 02551795966 mauroadame@hotmail.com
0704 3 Fabiana Alves
44999357396
Proprietário 92946100987 alves.fabiana@gmail.com
Washington Moretto Beraldo
4499047719
Residente 02123883980 wahsitberaldo@gmail.com
0705 3 Ademilson Aparecido Pereira
4430314139
Proprietário 62851110900 pereiradirect@hotmail.com
0706 3 Carlos Roberto Beleti Junior
4430256751 (4499739032)
Proprietário 05174731943 beleti.junior@gmail.com
Heitor Bonetti Santos Silva
44999721944
Residente 05069148958 heitorbonettisantossilva@gmail.com
0707 3 Ronyerison Henrique Boldrin
4432537127
Proprietário vendas6@duneyalimentos.com.br
Ariadnes Frias dos Santos
44991411613
Residente 06092828910 ariadnesfrias@gmail.com;jessicafslisboa@gmail.com
0708 3 Daniel Finkler de Almeida Proprietário 10103561943 daniel_finkler@hotmail.com
0801 3 Franciele da Silva Ferreira Zsigmond
4432447072
Proprietário 06410381948 frann_sf@hotmail.com;frannsfz@hotmail.com;
gzsm@hotmail.com.com
0802 3 Anne Lilian Marega
4432761269
Proprietário 05813625941 annemarega@yahoo.com
0803 3 João Paulo Pereira Vieira
4432250523 (44998445734)
Proprietário 09162834967 joh_vieira@live.com;kesia_paula@hotmail.com
0804 3 Glaucio Henrique Mazzo
4432620031 (4432601191)
Proprietário 06653253988 glaciomazzo@hotmail.com
0805 3 William Dias Soares Caetano
+55 1193018 71 88
Proprietário 05516329922 laidec.silva@yahoo.com.br
0806 3 Leandro Roberto de Souza Zerbinatti
4432465294 (44999538210)
Proprietário 03079907965 lrszerbinatti@yahoo.com.br
Marcelo Costa de Abreu
44997023642 (44997685086)
Residente 06811939990 rosana_kruger@outlook.com.br
0807 3 Pedro Prestes / Katia Maria Sanches Prestes
+55 43999756262
Proprietário 59887206920 thaise_prestes@hotmail.com
Estela Ladeira de Souza
+55 44999358653;+55 44999499820
Residente 07247276983 estelaladeira03@gmail.com;rd377336@gmail.com
0808 3 Claudio Rogerio Pereira Soares
4432683771
Proprietário 02549356993 claudio.rogerio76@gmail.com
"""

# Clean up header/footer noise and standardize markers
clean_text = ocr_text
clean_text = re.sub(r"Proprietário", "|Proprietário|", clean_text)
clean_text = re.sub(r"Residente", "|Residente|", clean_text)

# Split by the delimited markers to identify chunks
# The strategy: Split by '|'. Iterate. 
# If token is 'Proprietário' or 'Residente', the PREVIOUS token is the name/unit data, 
# and the NEXT token usually contains CPF/Email/Remainder.
tokens = clean_text.split('|')

data = []
current_unit = ""

# Iterate through tokens. 
# Structure is: [Data_Before] [TYPE] [Data_After] [Data_Before_Next] [TYPE] ...
# We look for the TYPE and merge the surrounding info.
for i in range(1, len(tokens), 2):
    contact_type = tokens[i] # Proprietário or Residente
    
    # 1. Process the text BEFORE the type (contains Unit, Name, Phones)
    pre_text = tokens[i-1].strip()
    
    # Check for unit at the start of this block
    unit_match = re.search(r'^(\d{4}\s\d)', pre_text)
    if unit_match:
        current_unit = unit_match.group(1)
        # Remove unit from text
        pre_text = pre_text[len(current_unit):].strip()
    
    # Extract phones from pre_text
    phones = []
    # Regex for various phone formats seen: (XX) XXXX-XXXX, +55..., or just digits
    phone_matches = re.findall(r'(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}[-]?\d{4}(?:[;/]\s*(?:\(?\d{2}\)?\s?)?\d{4,5}[-]?\d{4})*', pre_text)
    
    # Filter out things that look like CPF/CNPJ (11 or 14 digits continuous) from phone list just in case, 
    # though usually phones have hyphens or parens in this text.
    valid_phones = []
    for p in phone_matches:
        clean_p = re.sub(r'\D', '', p)
        if len(clean_p) < 11 or p.count('-') > 0 or p.count('(') > 0:
             valid_phones.append(p)
             pre_text = pre_text.replace(p, "")
    
    phones_str = "; ".join(valid_phones).strip()
    
    # Name is what remains in pre_text
    name = re.sub(r'\s+', ' ', pre_text).strip()
    
    # 2. Process the text AFTER the type (contains CPF, Email)
    # The 'post_text' goes until the end of the string or the beginning of the next unit logic.
    # However, in the split list, tokens[i+1] is the text between this Type and the next Type.
    # This implies tokens[i+1] contains the CPF/Email for THIS person, AND the Name/Unit for the NEXT person.
    # We need to split tokens[i+1] carefully.
    
    post_text = tokens[i+1] if i+1 < len(tokens) else ""
    
    # Find CPF (11 or 14 digits) usually appearing right after the type
    cpf = ""
    cpf_match = re.search(r'\b\d{11}\b|\b\d{14}\b', post_text)
    if cpf_match:
        cpf = cpf_match.group(0)
        # Remove from post_text so it doesn't interfere with next steps (though strictly not needed if we parse linearly)
    
    # Find Emails
    emails = []
    email_matches = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+(?:;[\w\.-]+@[\w\.-]+\.\w+)*', post_text)
    for email in email_matches:
        emails.append(email)
    email_str = "; ".join(emails)
    
    # Add to dataset
    data.append({
        'Unidade': current_unit,
        'Nome': name,
        'Telefone/Celular': phones_str,
        'Tipo': contact_type,
        'CPF/CNPJ': cpf,
        'E-mail': email_str
    })

# Create DataFrame
df = pd.DataFrame(data)

# Clean up Name column (remove trailing punctuation or noise)
df['Nome'] = df['Nome'].str.replace(r'[;]$', '', regex=True).str.strip()
# Clean up Unidade (ensure consistent spacing)
df['Unidade'] = df['Unidade'].str.replace(r'\s+', ' ', regex=True)

# Export to Excel
excel_path = 'Contatos_Residencial_Spazio_Montello.xlsx'
df.to_excel(excel_path, index=False)