emoji=(♈ ♉ ♊ ♋ ♌ ♎ ♏ ♐ ♑ ♒ ♓ ⛎ 🔯)
git add -A
echo "###Add Finish"
git commit -am "${emoji[$(($RANDOM % ${#emoji[@]} + 1 ))]} xixici  push at $(date "+%Y-%m-%d %H:%M:%S")"
echo "###Commit Finish"
git pull
echo "###Pull Finish"
git status
echo "###Status Finish"
git push
echo "###push Finish"
