#!/bin/bash

for VARIABLE in traducir traduzco traduces traduce traduzca traduzcas
do
	echo " $VARIABLE appear :"$(grep -wc $VARIABLE vocab.json)
done
