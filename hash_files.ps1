$files = Get-ChildItem -Path 'C:\Users\jimjj\Music\QuantumChildren' -Recurse -Filter '*.mq5' | Where-Object { $_.FullName -notmatch 'my-trading-work|graveyard' }
foreach($f in $files) {
    $hash = (Get-FileHash $f.FullName -Algorithm MD5).Hash
    Write-Output ('{0}|{1}|{2}' -f $hash, $f.Length, $f.FullName)
}
